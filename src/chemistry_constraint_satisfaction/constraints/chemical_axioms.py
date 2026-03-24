"""
Chemical constraint checks.

The supervisor uses these helpers to verify candidates against a few
simple rules (mass, charge, and bond valency). When `z3` is available we
can run the same checks with a solver; otherwise we fall back to a pure
Python implementation.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional Z3 import — graceful fallback for environments without the solver
# ---------------------------------------------------------------------------
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    z3 = None  # type: ignore
    Z3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Element data  (Z = 1 .. 86, H through Rn)
# IUPAC 2021 standard atomic weights; max common covalent valencies.
# ---------------------------------------------------------------------------

#: Standard atomic masses (u).
ATOMIC_MASS: Dict[str, float] = {
    "H": 1.008, "He": 4.003,
    "Li": 6.941, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922,
    "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.95, "Tc": 97.0, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42,
    "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760,
    "Te": 127.60, "I": 126.904, "Xe": 131.293,
    "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Pm": 145.0,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500,
    "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967,
    "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207, "Os": 190.23,
    "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.980, "Po": 209.0, "At": 210.0,
    "Rn": 222.0,
}

#: Maximum common covalent valency for each element.
MAX_VALENCY: Dict[str, int] = {
    "H": 1, "He": 0,
    "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Ne": 0,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 0,
    "K": 1, "Ca": 2, "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7,
    "Fe": 6, "Co": 6, "Ni": 4, "Cu": 4, "Zn": 2,
    "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 2,
    "Rb": 1, "Sr": 2, "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7,
    "Ru": 8, "Rh": 6, "Pd": 4, "Ag": 3, "Cd": 2,
    "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,
    "Cs": 1, "Ba": 2,
    "La": 3, "Ce": 4, "Pr": 4, "Nd": 3, "Pm": 3,
    "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 4, "Dy": 3,
    "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8,
    "Ir": 6, "Pt": 6, "Au": 5, "Hg": 4,
    "Tl": 3, "Pb": 4, "Bi": 5, "Po": 6, "At": 7, "Rn": 2,
}

#: Atomic number for each element symbol.
ATOMIC_NUMBER: Dict[str, int] = {
    sym: z for z, sym in enumerate([
        None, "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn",
    ], start=0) if sym is not None
}

#: Extra valency allowed when an atom has a formal charge.
CHARGE_VALENCY_DELTA: Dict[str, Dict[int, int]] = {
    "H":  {-1: -1},
    "Li": {+1: -1},
    "B":  {-1: +1},
    "C":  {-1: -1, +1: -1},
    "N":  {+1: +1, -1: -1},
    "O":  {-1: -1, +1: +1},
    "F":  {-1: -1},
    "Na": {+1: -1},
    "Mg": {+2: -2},
    "Al": {+3: -3},
    "Si": {-1: +1},
    "P":  {+1: +1},
    "S":  {+1: +1, +2: +2, -1: -1},
    "Cl": {-1: -1, +1: +1},
    "K":  {+1: -1},
    "Ca": {+2: -2},
    "Fe": {+2: -2, +3: -3},
    "Cu": {+1: -1, +2: -2},
    "Zn": {+2: -2},
    "Br": {-1: -1, +1: +1},
    "Ag": {+1: -1},
    "Sn": {+2: -2, +4: -4},
    "I":  {-1: -1, +1: +1},
    "Pt": {+2: -2, +4: -4},
    "Au": {+1: -1, +3: -3},
    "Hg": {+1: -1, +2: -2},
    "Pb": {+2: -2, +4: -4},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Atom:
    """Lightweight atom representation used by the constraint engine."""
    element: str
    bonds: int            # total bond order to other atoms
    formal_charge: int = 0
    implicit_h: int = 0   # implicit hydrogen count

    @property
    def effective_valency(self) -> int:
        base = MAX_VALENCY.get(self.element, 4)
        delta_map = CHARGE_VALENCY_DELTA.get(self.element, {})
        return base + delta_map.get(self.formal_charge, 0)

    @property
    def total_bonds(self) -> int:
        """Bonds to heavy atoms + implicit hydrogens."""
        return self.bonds + self.implicit_h


@dataclasses.dataclass
class MolecularState:
    """Snapshot of a (partial) molecule during diffusion."""
    name: str
    atoms: List[Atom]

    def total_mass(self) -> float:
        mass = 0.0
        for atom in self.atoms:
            mass += ATOMIC_MASS.get(atom.element, 0.0)
            mass += atom.implicit_h * ATOMIC_MASS["H"]
        return mass

    def total_charge(self) -> int:
        return sum(a.formal_charge for a in self.atoms)


@dataclasses.dataclass
class ConstraintResult:
    sat: bool
    violations: List[str] = dataclasses.field(default_factory=list)
    z3_model: Optional[object] = None  # z3.ModelRef when available

    @property
    def reason(self) -> str:
        if self.sat:
            return "All constraints satisfied."
        return "; ".join(self.violations)

    def __bool__(self) -> bool:
        return self.sat


# ---------------------------------------------------------------------------
# Pure-Python fallback checker (no Z3 required)
# ---------------------------------------------------------------------------

def _check_pure_python(
    reactants: List[MolecularState],
    products: List[MolecularState],
    tolerance: float = 0.02,
) -> ConstraintResult:
    """Check constraints without Z3."""
    violations: List[str] = []

    # 1. Mass conservation
    r_mass = sum(m.total_mass() for m in reactants)
    p_mass = sum(m.total_mass() for m in products)
    if abs(r_mass - p_mass) > tolerance:
        violations.append(
            f"Mass not conserved: reactants={r_mass:.3f} u, "
            f"products={p_mass:.3f} u, Δ={abs(r_mass-p_mass):.3f} u"
        )

    # 2. Charge conservation
    r_charge = sum(m.total_charge() for m in reactants)
    p_charge = sum(m.total_charge() for m in products)
    if r_charge != p_charge:
        violations.append(
            f"Charge not conserved: reactants={r_charge:+d}, "
            f"products={p_charge:+d}"
        )

    # 3. Bond valency for each product atom
    for mol in products:
        for atom in mol.atoms:
            if atom.total_bonds > atom.effective_valency:
                violations.append(
                    f"{mol.name}: {atom.element} has {atom.total_bonds} bonds "
                    f"(max {atom.effective_valency})"
                )

    return ConstraintResult(sat=len(violations) == 0, violations=violations)


# ---------------------------------------------------------------------------
# Z3-backed checker
# ---------------------------------------------------------------------------

def _check_z3(
    reactants: List[MolecularState],
    products: List[MolecularState],
    tolerance: float = 0.02,
) -> ConstraintResult:
    """
    Check constraints using Z3.

    This version uses real arithmetic for mass and integer arithmetic for
    charge and valency.
    """
    if not Z3_AVAILABLE:
        raise RuntimeError("z3 is not installed; use _check_pure_python instead.")

    solver = z3.Solver()
    violations: List[str] = []

    # Set up Z3 constraints. We treat the provided values as fixed facts and
    # ask whether they violate the valency bounds.

    # Mass conservation (Real arithmetic)
    r_mass_expr = z3.RealVal(0)
    for mol in reactants:
        for atom in mol.atoms:
            r_mass_expr = r_mass_expr + z3.RealVal(ATOMIC_MASS.get(atom.element, 0.0))
            r_mass_expr = r_mass_expr + z3.RealVal(atom.implicit_h) * z3.RealVal(ATOMIC_MASS["H"])

    p_mass_expr = z3.RealVal(0)
    for mol in products:
        for atom in mol.atoms:
            p_mass_expr = p_mass_expr + z3.RealVal(ATOMIC_MASS.get(atom.element, 0.0))
            p_mass_expr = p_mass_expr + z3.RealVal(atom.implicit_h) * z3.RealVal(ATOMIC_MASS["H"])

    mass_diff = z3.simplify(r_mass_expr - p_mass_expr)
    tol = z3.RealVal(tolerance)
    solver.add(z3.Or(mass_diff > tol, mass_diff < -tol))  # UNSAT means conserved

    result_mass = solver.check()
    solver.reset()
    if result_mass == z3.sat:
        # If the "violation" query is satisfiable, the masses differ too much.
        from fractions import Fraction
        r_val = float(Fraction(str(z3.simplify(r_mass_expr))))
        p_val = float(Fraction(str(z3.simplify(p_mass_expr))))
        violations.append(
            f"Mass not conserved: reactants≈{r_val:.3f} u, products≈{p_val:.3f} u"
        )

    # Charge conservation (Int arithmetic)
    r_chg = sum(sum(a.formal_charge for a in m.atoms) for m in reactants)
    p_chg = sum(sum(a.formal_charge for a in m.atoms) for m in products)
    if r_chg != p_chg:
        violations.append(
            f"Charge not conserved: reactants={r_chg:+d}, products={p_chg:+d}"
        )

    # Valency per atom (Int arithmetic)
    for mol in products:
        for i, atom in enumerate(mol.atoms):
            bonds_var = z3.Int(f"{mol.name}_{atom.element}_{i}_bonds")
            max_val   = z3.IntVal(atom.effective_valency)
            solver.add(bonds_var == z3.IntVal(atom.total_bonds))
            solver.add(bonds_var > max_val)          # ask if valency is exceeded
            if solver.check() == z3.sat:
                violations.append(
                    f"{mol.name}: {atom.element}[{i}] has {atom.total_bonds} bonds "
                    f"(max {atom.effective_valency} for charge {atom.formal_charge:+d})"
                )
            solver.reset()

    return ConstraintResult(sat=len(violations) == 0, violations=violations)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_reaction(
    reactants: List[MolecularState],
    products: List[MolecularState],
    tolerance: float = 0.02,
    prefer_z3: bool = True,
) -> ConstraintResult:
    """
    Verify that a proposed reaction satisfies all chemical axioms.

    Parameters
    ----------
    reactants   : molecules before the reaction
    products    : molecules after the reaction (may be partial / intermediate)
    tolerance   : mass tolerance in atomic mass units (default 0.02 u)
    prefer_z3   : use Z3 when available; fall back to pure-Python otherwise

    Returns
    -------
    ConstraintResult with .sat, .violations, and .reason
    """
    if prefer_z3 and Z3_AVAILABLE:
        return _check_z3(reactants, products, tolerance)
    return _check_pure_python(reactants, products, tolerance)


def check_intermediate(
    molecule: MolecularState,
    tolerance: float = 0.02,
) -> ConstraintResult:
    """
    Check a single intermediate molecule's internal consistency
    (valency only — mass conservation requires reactant context).
    """
    violations: List[str] = []
    for i, atom in enumerate(molecule.atoms):
        if atom.total_bonds > atom.effective_valency:
            violations.append(
                f"{molecule.name}: {atom.element}[{i}] has "
                f"{atom.total_bonds} bonds (max {atom.effective_valency})"
            )
    return ConstraintResult(sat=len(violations) == 0, violations=violations)
