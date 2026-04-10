"""
Supervisor loop -- wraps the diffusion model and enforces chemistry
constraints at every reverse step (commit / fix / backtrack).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Tuple

import numpy as np

from ..constraints.chemical_axioms import (
    MolecularState, ConstraintResult, check_intermediate, check_reaction,
    ATOMIC_MASS,
)
from .model import MolecularDiffusionModel, encode_molecule


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class StepRecord:
    t: int
    attempt: int
    constraint_result: ConstraintResult
    action: str   # "commit", "corrected", "backtrack", "skip"
    elapsed_ms: float


@dataclasses.dataclass
class GenerationResult:
    product: MolecularState
    reactants: List[MolecularState]
    final_check: ConstraintResult
    step_log: List[StepRecord]
    total_backtracks: int
    total_corrections: int
    wall_time_s: float

    @property
    def success(self) -> bool:
        return self.final_check.sat

    @property
    def metrics(self) -> Dict[str, float]:
        n_steps = len(self.step_log) or 1
        commits = sum(1 for s in self.step_log if s.action == "commit")
        corrected = sum(1 for s in self.step_log if s.action == "corrected")
        skips = sum(1 for s in self.step_log if s.action == "skip")
        avg_ms = sum(s.elapsed_ms for s in self.step_log) / n_steps
        return {
            "valid": float(self.success),
            "commits": commits,
            "corrections": corrected,
            "backtracks": self.total_backtracks,
            "skips": skips,
            "avg_step_ms": round(avg_ms, 2),
            "wall_time_s": round(self.wall_time_s, 3),
            "commit_rate": round(commits / n_steps, 3),
        }

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=" * 60,
            "  Supervisor - Generation Summary",
            "=" * 60,
            f"  Product      : {self.product.name}",
            f"  Atoms        : {len(self.product.atoms)}",
            f"  Valid        : {'YES' if self.success else 'NO'}",
            f"  Backtracks   : {self.total_backtracks}",
            f"  Corrections  : {self.total_corrections}",
            f"  Commit rate  : {m['commit_rate']:.1%}",
            f"  Avg step     : {m['avg_step_ms']:.1f} ms",
            f"  Wall time    : {self.wall_time_s:.3f}s",
            "",
            "  Constraint check:",
        ]
        if self.success:
            lines.append("    [OK] All axioms satisfied.")
        else:
            for v in self.final_check.violations:
                lines.append(f"    [FAIL] {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Correction strategies
# ---------------------------------------------------------------------------

def _fix_valency(mol: MolecularState) -> MolecularState:
    fixed_atoms = []
    for atom in mol.atoms:
        if atom.total_bonds > atom.effective_valency:
            excess = atom.total_bonds - atom.effective_valency
            new_bonds  = max(0, atom.bonds - excess)
            new_impl_h = max(0, atom.implicit_h - max(0, excess - atom.bonds))
            fixed_atoms.append(dataclasses.replace(atom, bonds=new_bonds, implicit_h=new_impl_h))
        else:
            fixed_atoms.append(atom)
    return MolecularState(atoms=fixed_atoms, name=mol.name)


def _fix_mass(
    mol: MolecularState,
    target_mass: float,
    tolerance: float = 0.02,
) -> MolecularState:
    from ..constraints.chemical_axioms import ATOMIC_MASS
    current = mol.total_mass()
    delta   = target_mass - current
    h_mass  = ATOMIC_MASS["H"]

    if abs(delta) <= tolerance:
        return mol

    # Distribute hydrogen adjustments across atoms
    atoms = [dataclasses.replace(a) for a in mol.atoms]
    for atom in atoms:
        if abs(delta) <= tolerance:
            break
        if delta > 0:  # need more mass → add H
            add = min(int(delta / h_mass), atom.effective_valency - atom.total_bonds)
            atom.implicit_h += add
            delta -= add * h_mass
        else:           # need less mass → remove H
            remove = min(int(-delta / h_mass), atom.implicit_h)
            atom.implicit_h -= remove
            delta += remove * h_mass

    return MolecularState(atoms=atoms, name=mol.name)


def _fix_charge(
    mol: MolecularState,
    target_charge: int,
) -> MolecularState:
    """Shift formal charges (lightest atoms first) until net charge == target.
    Pretty crude heuristic but better than doing nothing."""
    current = mol.total_charge()
    delta = target_charge - current
    if delta == 0:
        return mol

    atoms = [dataclasses.replace(a) for a in mol.atoms]
    sorted_idx = sorted(range(len(atoms)),
                        key=lambda i: ATOMIC_MASS.get(atoms[i].element, 999))

    for idx in sorted_idx:
        if delta == 0:
            break
        step = 1 if delta > 0 else -1
        atoms[idx].formal_charge += step
        delta -= step

    return MolecularState(atoms=atoms, name=mol.name)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:
    """Runs the diffusion model with chemistry checks at each step."""

    def __init__(
        self,
        model: MolecularDiffusionModel,
        reactants: List[MolecularState],
        T: int = 50,
        max_retries: int = 3,
        max_backtracks: int = 10,
        verbose: bool = False,
        prefer_z3: bool = True,
    ):
        self.model          = model
        self.reactants      = reactants
        self.T              = T
        self.max_retries    = max_retries
        self.max_backtracks = max_backtracks
        self.verbose        = verbose
        self.prefer_z3      = prefer_z3

        self._target_mass   = sum(m.total_mass()   for m in reactants)
        self._target_charge = sum(m.total_charge() for m in reactants)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> GenerationResult:
        t0 = time.perf_counter()
        step_log: List[StepRecord] = []
        total_backtracks = 0
        total_corrections = 0

        init_mol = self._build_initial_state()
        x, adj   = encode_molecule(init_mol)
        x_t, adj_t = self.model.forward_noisy(x, adj, self.T, self.T)

        history: List[Tuple[np.ndarray, np.ndarray]] = [(x_t.copy(), adj_t.copy())]
        t = self.T
        while t >= 1:
            step_start = time.perf_counter()
            committed  = False
            cr = None

            for attempt in range(1, self.max_retries + 2):
                x_prev, adj_prev = self.model.reverse_step(x_t, adj_t, t, self.T)
                candidate = self.model.decode(x_prev, adj_prev, name="intermediate")

                cr = check_intermediate(candidate)

                if t == 1:  # final step -- full conservation check
                    candidate = MolecularState(
                        name="product", atoms=candidate.atoms,
                    )
                    cr = check_reaction(
                        self.reactants, [candidate],
                        prefer_z3=self.prefer_z3,
                    )

                elapsed = (time.perf_counter() - step_start) * 1000

                if cr.sat:
                    action = "commit" if attempt == 1 else "corrected"
                    step_log.append(StepRecord(t, attempt, cr, action, elapsed))
                    x_t, adj_t = x_prev, adj_prev
                    history.append((x_t.copy(), adj_t.copy()))
                    committed = True
                    if attempt > 1:
                        total_corrections += 1
                    if self.verbose:
                        self._log(t, action, cr, elapsed)
                    break
                else:
                    if attempt <= self.max_retries:
                        candidate = _fix_valency(candidate)
                        if t == 1:
                            candidate = _fix_charge(candidate, self._target_charge)
                            candidate = _fix_mass(candidate, self._target_mass)
                        cr2 = (
                            check_reaction(self.reactants, [candidate], prefer_z3=self.prefer_z3)
                            if t == 1 else check_intermediate(candidate)
                        )
                        if cr2.sat:
                            x_prev, adj_prev = encode_molecule(candidate)
                            x_t, adj_t = x_prev, adj_prev
                            history.append((x_t.copy(), adj_t.copy()))
                            committed = True
                            total_corrections += 1
                            step_log.append(StepRecord(t, attempt, cr2, "corrected", elapsed))
                            if self.verbose:
                                self._log(t, "corrected", cr2, elapsed)
                            break

            if not committed:
                # Backtrack
                total_backtracks += 1
                if self.verbose:
                    print(f"  [t={t:3d}] BACKTRACK #{total_backtracks} -- violations: {cr.reason}")
                if len(history) > 1:
                    failed_t = t
                    history.pop()
                    x_t, adj_t = history[-1]
                    t = min(t + 1, self.T)
                    step_log.append(StepRecord(failed_t, 0, cr, "backtrack", 0.0))
                else:
                    step_log.append(StepRecord(t, 0, cr, "skip", 0.0))
                    x_t, adj_t = x_prev, adj_prev
                    history.append((x_t.copy(), adj_t.copy()))

                if total_backtracks >= self.max_backtracks:
                    if self.verbose:
                        print(f"  [supervisor] Max backtracks reached; stopping early.")
                    break

            t -= 1

        product = self.model.decode(x_t, adj_t, name="product")
        final_check = check_reaction(
            self.reactants, [product], prefer_z3=self.prefer_z3
        )

        wall_time = time.perf_counter() - t0
        result = GenerationResult(
            product=product,
            reactants=self.reactants,
            final_check=final_check,
            step_log=step_log,
            total_backtracks=total_backtracks,
            total_corrections=total_corrections,
            wall_time_s=wall_time,
        )

        if self.verbose:
            print(result.summary())

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> MolecularState:
        atoms = []
        for mol in self.reactants:
            atoms.extend(mol.atoms)
        return MolecularState(name="reactants_concat", atoms=atoms)

    def _log(self, t: int, action: str, cr: ConstraintResult, elapsed: float) -> None:
        icon = "[OK]" if cr.sat else "[FAIL]"
        print(f"  [t={t:3d}] {icon} {action:<12} ({elapsed:.1f} ms)")
        if not cr.sat:
            for v in cr.violations:
                print(f"           -> {v}")
