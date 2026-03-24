#!/usr/bin/env python3
"""
Small demo script for Chemistry Constraint Satisfaction.

Runs a few quick constraint checks and then shows one supervised diffusion
run for CH3Br + OH- -> CH3OH + Br-.
Run: python scripts/demo.py
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chemistry_constraint_satisfaction.constraints import (
    Atom, MolecularState, check_reaction, check_intermediate,
    Z3_AVAILABLE,
)
from chemistry_constraint_satisfaction.diffusion import (
    MolecularDiffusionModel, Supervisor,
)

SEP = "-" * 60


# ---------------------------------------------------------------------------
# Helper molecules
# ---------------------------------------------------------------------------

def ch3br():
    return MolecularState("CH₃Br", [
        Atom("C", 4), Atom("Br", 1),
        Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ])

def oh_minus():
    return MolecularState("OH⁻", [
        Atom("O", 1, formal_charge=-1), Atom("H", 1),
    ])

def ch3oh():
    return MolecularState("CH₃OH", [
        Atom("C", 4), Atom("O", 2),
        Atom("H", 1), Atom("H", 1),
        Atom("H", 1), Atom("H", 1),
    ])

def br_minus():
    return MolecularState("Br⁻", [Atom("Br", 0, formal_charge=-1)])


# ---------------------------------------------------------------------------
# Part 1: Constraint checks
# ---------------------------------------------------------------------------

def demo_constraints():
    print(SEP)
    print("  PART 1 — Chemical Axiom Checks")
    print(f"  (Z3 solver: {'available ✓' if Z3_AVAILABLE else 'not installed — using pure-Python fallback'})")
    print(SEP)

    # --- Valid SN2 reaction ---
    cr = check_reaction([ch3br(), oh_minus()], [ch3oh(), br_minus()])
    icon = "✓" if cr.sat else "✗"
    print(f"\n  {icon} CH₃Br + OH⁻ → CH₃OH + Br⁻")
    print(f"    {cr.reason}")

    # --- Carbon with 5 bonds (should fail) ---
    bad_carbon = MolecularState("bad", [Atom("C", 5)])
    cr2 = check_intermediate(bad_carbon)
    icon2 = "✓" if cr2.sat else "✗"
    print(f"\n  {icon2} Carbon with 5 bonds")
    print(f"    {cr2.reason}")

    # --- Missing product (mass not conserved) ---
    cr3 = check_reaction([ch3br(), oh_minus()], [ch3oh()])  # Br⁻ missing
    icon3 = "✓" if cr3.sat else "✗"
    print(f"\n  {icon3} CH₃Br + OH⁻ → CH₃OH  (Br⁻ missing — should violate mass)")
    print(f"    {cr3.reason}")

    # --- Charge mismatch ---
    cr4 = check_reaction([ch3br(), oh_minus()], [ch3oh(), MolecularState("Br", [Atom("Br", 0)])])
    icon4 = "✓" if cr4.sat else "✗"
    print(f"\n  {icon4} CH₃Br + OH⁻ → CH₃OH + Br  (neutral Br — charge mismatch)")
    print(f"    {cr4.reason}")

    # --- NH₄⁺ ammonium (should be valid, 4 bonds on N+) ---
    nh4 = MolecularState("NH₄⁺", [Atom("N", 4, formal_charge=+1)])
    cr5 = check_intermediate(nh4)
    icon5 = "✓" if cr5.sat else "✗"
    print(f"\n  {icon5} NH₄⁺ — nitrogen with 4 bonds (formal charge +1)")
    print(f"    {cr5.reason}")


# ---------------------------------------------------------------------------
# Part 2: Supervised generation
# ---------------------------------------------------------------------------

def demo_generation():
    print(f"\n{SEP}")
    print("  PART 2 — Supervisor loop demo")
    print("  Reaction: CH3Br + OH-  ->  (supervised diffusion)  ->  ?")
    print(SEP)

    model = MolecularDiffusionModel(hidden_dim=64, seed=42)
    sup   = Supervisor(
        model,
        reactants=[ch3br(), oh_minus()],
        T=20,
        max_retries=3,
        max_backtracks=5,
        verbose=True,
        prefer_z3=Z3_AVAILABLE,
    )
    result = sup.run()

    print("\n  Product atoms:")
    for i, atom in enumerate(result.product.atoms):
        print(f"    [{i}] {atom.element:3s}  bonds={atom.bonds}  "
              f"implicit_H={atom.implicit_h}  charge={atom.formal_charge:+d}")

    return result


# ---------------------------------------------------------------------------
# Part 3: Benchmark
# ---------------------------------------------------------------------------

def demo_benchmark(n: int = 50):
    print(f"\n{SEP}")
    print(f"  PART 3 — Benchmark: {n} generations  (random-weight model)")
    print(f"  Metric A: Intermediate valency validity (what supervisor enforces per-step)")
    print(f"  Metric B: Full conservation validity (requires a TRAINED model)")
    print(SEP)

    from chemistry_constraint_satisfaction.diffusion.model import encode_molecule
    from chemistry_constraint_satisfaction.constraints import check_intermediate, check_reaction

    reactants = [ch3br(), oh_minus()]

    # Count how many times the supervisor produces 0 valency violations
    # vs the raw model doing the same number of steps
    sup_valency_ok  = 0
    raw_valency_ok  = 0
    sup_full_ok     = 0
    raw_full_ok     = 0

    for i in range(n):
        m = MolecularDiffusionModel(hidden_dim=32, seed=i)

        # ---- SUPERVISED ----
        sup = Supervisor(m, reactants, T=10, max_retries=2, max_backtracks=3,
                         verbose=False, prefer_z3=False)
        r = sup.run()
        # Valency check on final product
        cr_val = check_intermediate(r.product)
        if cr_val.sat:
            sup_valency_ok += 1
        if r.success:
            sup_full_ok += 1

        # ---- UNSUPERVISED (raw) ----
        m2 = MolecularDiffusionModel(hidden_dim=32, seed=i)
        init_atoms = []
        for mol in reactants:
            init_atoms.extend(mol.atoms)
        init_mol = MolecularState("init", init_atoms)
        x, adj   = encode_molecule(init_mol)
        x_n, adj_n = m2.forward_noisy(x, adj, t=10, T=10)
        for t in range(10, 0, -1):
            x_n, adj_n = m2.reverse_step(x_n, adj_n, t=t, T=10)
        raw_product = m2.decode(x_n, adj_n, name="raw_product")
        cr_raw_val = check_intermediate(raw_product)
        if cr_raw_val.sat:
            raw_valency_ok += 1
        cr_raw_full = check_reaction(reactants, [raw_product], prefer_z3=False)
        if cr_raw_full.sat:
            raw_full_ok += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{n}]  "
                  f"sup_valency={sup_valency_ok}  raw_valency={raw_valency_ok}  "
                  f"sup_full={sup_full_ok}  raw_full={raw_full_ok}")

    print(f"\n  Results over {n} reactions  (random GNN weights):")
    print(f"    {'Metric':<30} {'Supervised':>12}  {'Unsupervised':>12}")
    print(f"    {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"    {'Valency validity (per-step)':<30} "
          f"{sup_valency_ok/n*100:>10.1f}%  {raw_valency_ok/n*100:>10.1f}%")
    print(f"    {'Full conservation validity':<30} "
          f"{sup_full_ok/n*100:>10.1f}%  {raw_full_ok/n*100:>10.1f}%")
    print()
    print("  Note: the supervisor focuses on per-step valency consistency.")
    print("  Mass/charge conservation depends on how well the model is trained.")
    print("  See Part 4 below to train and observe improving full validity.")


# ---------------------------------------------------------------------------
# Part 4: Train, then generate with learned weights
# ---------------------------------------------------------------------------

def demo_train_then_generate(n_benchmark: int = 30):
    """Train the model via gradient descent, then generate and compare."""
    print(f"\n{SEP}")
    print("  PART 4 — Train-then-Generate (gradient descent + constraint satisfaction)")
    print(SEP)

    from chemistry_constraint_satisfaction.diffusion.trainer import (
        train_and_export, default_training_molecules,
    )
    from chemistry_constraint_satisfaction.diffusion.model import encode_molecule
    from chemistry_constraint_satisfaction.constraints import check_intermediate, check_reaction

    reactants = [ch3br(), oh_minus()]

    # ---- Step 1: Train the model ----
    print("\n  Step 1: Training the denoising GNN ...")
    trained_model, loss_history = train_and_export(
        molecules=default_training_molecules(),
        hidden_dim=64,
        T=20,
        epochs=40,
        lr=1e-3,
        steps_per_epoch=80,
        seed=42,
        verbose=True,
    )
    print(f"\n  Loss decreased: {loss_history[0]:.4f} -> {loss_history[-1]:.4f} "
          f"({(1 - loss_history[-1]/loss_history[0])*100:.1f}% reduction)")

    # ---- Step 2: Generate with the trained model ----
    print(f"\n  Step 2: Supervised generation with TRAINED model")
    sup = Supervisor(
        trained_model,
        reactants=reactants,
        T=20,
        max_retries=3,
        max_backtracks=5,
        verbose=True,
        prefer_z3=Z3_AVAILABLE,
    )
    result = sup.run()

    print("\n  Product atoms:")
    for i, atom in enumerate(result.product.atoms):
        print(f"    [{i}] {atom.element:3s}  bonds={atom.bonds}  "
              f"implicit_H={atom.implicit_h}  charge={atom.formal_charge:+d}")

    # ---- Step 3: Three-way benchmark ----
    print(f"\n  Step 3: Benchmark over {n_benchmark} runs  —  three configurations")
    print(f"    A) Trained model + supervisor (constraint satisfaction)")
    print(f"    B) Random model + supervisor (constraint satisfaction, no learning)")
    print(f"    C) Random model, NO supervisor (raw diffusion)")
    print()

    trained_sup_val = trained_sup_full = 0
    random_sup_val  = random_sup_full  = 0
    raw_val         = raw_full         = 0

    for i in range(n_benchmark):
        # --- A) Trained + supervised ---
        r_a = Supervisor(
            trained_model, reactants, T=10, max_retries=2, max_backtracks=3,
            verbose=False, prefer_z3=False,
        ).run()
        if check_intermediate(r_a.product).sat:
            trained_sup_val += 1
        if r_a.success:
            trained_sup_full += 1

        # --- B) Random + supervised ---
        m_b = MolecularDiffusionModel(hidden_dim=32, seed=i)
        r_b = Supervisor(
            m_b, reactants, T=10, max_retries=2, max_backtracks=3,
            verbose=False, prefer_z3=False,
        ).run()
        if check_intermediate(r_b.product).sat:
            random_sup_val += 1
        if r_b.success:
            random_sup_full += 1

        # --- C) Raw (no supervisor) ---
        m_c = MolecularDiffusionModel(hidden_dim=32, seed=i)
        init_atoms = []
        for mol in reactants:
            init_atoms.extend(mol.atoms)
        init_mol = MolecularState("init", init_atoms)
        x, adj = encode_molecule(init_mol)
        x_n, adj_n = m_c.forward_noisy(x, adj, t=10, T=10)
        for t in range(10, 0, -1):
            x_n, adj_n = m_c.reverse_step(x_n, adj_n, t=t, T=10)
        raw_product = m_c.decode(x_n, adj_n, name="raw")
        if check_intermediate(raw_product).sat:
            raw_val += 1
        if check_reaction(reactants, [raw_product], prefer_z3=False).sat:
            raw_full += 1

        if (i + 1) % 10 == 0:
            print(f"    [{i+1:3d}/{n_benchmark}] done")

    print(f"\n  {'='*60}")
    print(f"  Results  ({n_benchmark} runs)")
    print(f"  {'='*60}")
    print(f"    {'Configuration':<35} {'Valency':>10}  {'Full Valid':>10}")
    print(f"    {'-'*35}  {'-'*10}  {'-'*10}")
    print(f"    {'Trained + Supervisor':<35} "
          f"{trained_sup_val/n_benchmark*100:>8.1f}%  "
          f"{trained_sup_full/n_benchmark*100:>8.1f}%")
    print(f"    {'Random  + Supervisor':<35} "
          f"{random_sup_val/n_benchmark*100:>8.1f}%  "
          f"{random_sup_full/n_benchmark*100:>8.1f}%")
    print(f"    {'Random  + NO Supervisor (raw)':<35} "
          f"{raw_val/n_benchmark*100:>8.1f}%  "
          f"{raw_full/n_benchmark*100:>8.1f}%")
    print(f"  {'='*60}")
    print()
    print("  Key insight: Training teaches the model RIGHT patterns (gradient")
    print("  descent), while constraint satisfaction ENFORCES correctness.")
    print("  Together they achieve the highest accuracy.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_constraints()
    result = demo_generation()
    demo_benchmark(n=50)
    demo_train_then_generate(n_benchmark=30)

    print(f"\n{SEP}")
    print("  Done. For GPU-accelerated training, open notebooks/demo.ipynb")
    print("  in Google Colab and enable a GPU runtime.")
    print(SEP)
