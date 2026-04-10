#!/usr/bin/env python3
"""Generate figures for the final report."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState, check_intermediate, check_reaction,
)
from chemistry_constraint_satisfaction.diffusion.training import (
    train, TrainConfig, export_to_numpy,
)
from chemistry_constraint_satisfaction.diffusion.model import (
    MolecularDiffusionModel, encode_molecule,
)
from chemistry_constraint_satisfaction.diffusion.supervisor import Supervisor

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------- training data ----------
water = MolecularState("H2O", [Atom("O", 2), Atom("H", 1), Atom("H", 1)])
methane = MolecularState("CH4", [
    Atom("C", 4), Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
])
ethanol = MolecularState("C2H5OH", [
    Atom("C", 4), Atom("C", 4), Atom("O", 2),
    Atom("H", 1), Atom("H", 1), Atom("H", 1),
    Atom("H", 1), Atom("H", 1), Atom("H", 1),
])

sn2_r = [
    MolecularState("CH3Br", [
        Atom("C", 4), Atom("Br", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("OH-", [Atom("O", 1, formal_charge=-1), Atom("H", 1)]),
]
comb_r = [
    MolecularState("CH4", [
        Atom("C", 4), Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("O2_a", [Atom("O", 2), Atom("O", 2)]),
    MolecularState("O2_b", [Atom("O", 2), Atom("O", 2)]),
]
acid_r = [
    MolecularState("HCl", [Atom("H", 1), Atom("Cl", 1)]),
    MolecularState("NaOH", [Atom("O", 2), Atom("H", 1)]),
]


# =====================================================================
# Figure 1: Training loss curve
# =====================================================================
print("Training model for loss curve...")
cfg = TrainConfig(lr=1e-3, epochs=200, hidden_dim=32, num_layers=2,
                  log_every=50, device="cpu")
result = train([water, methane, ethanol], config=cfg, verbose=True)

epochs = list(range(1, cfg.epochs + 1))
totals = [d["total"] for d in result.epoch_losses]
denoise = [d["denoising"] for d in result.epoch_losses]
bond = [d["bond_ce"] for d in result.epoch_losses]
val_pen = [d["valency_penalty"] for d in result.epoch_losses]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, totals, label="Total loss", linewidth=2)
ax.plot(epochs, denoise, label="Denoising (MSE)", linewidth=1, alpha=0.7)
ax.plot(epochs, bond, label="Bond CE", linewidth=1, alpha=0.7)
ax.plot(epochs, val_pen, label="Valency penalty", linewidth=1, alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Over 200 Epochs (3-molecule dataset)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_training_loss.png"), dpi=150)
print(f"  -> {OUT}/fig1_training_loss.png")
plt.close()


# =====================================================================
# Figure 2: Supervised vs Unsupervised valency validity (bar chart)
# =====================================================================
print("Running benchmark for bar chart...")
n = 50
reactions = {
    "SN2\n(CH₃Br + OH⁻)": sn2_r,
    "Combustion\n(CH₄ + 2O₂)": comb_r,
    "Acid-base\n(HCl + NaOH)": acid_r,
}

sup_vals = []
raw_vals = []
for name, reactants in reactions.items():
    s_ok, r_ok = 0, 0
    for i in range(n):
        m = MolecularDiffusionModel(hidden_dim=32, seed=i)
        # supervised
        sup = Supervisor(m, reactants, T=10, max_retries=2, max_backtracks=3,
                         verbose=False, prefer_z3=False)
        res = sup.run()
        if check_intermediate(res.product).sat:
            s_ok += 1
        # unsupervised
        m2 = MolecularDiffusionModel(hidden_dim=32, seed=i)
        atoms = []
        for mol in reactants:
            atoms.extend(mol.atoms)
        init = MolecularState("init", atoms)
        x, adj = encode_molecule(init)
        xn, an = m2.forward_noisy(x, adj, t=10, T=10)
        for t in range(10, 0, -1):
            xn, an = m2.reverse_step(xn, an, t=t, T=10)
        raw = m2.decode(xn, an, name="raw")
        if check_intermediate(raw).sat:
            r_ok += 1
    sup_vals.append(s_ok / n * 100)
    raw_vals.append(r_ok / n * 100)

labels = list(reactions.keys())
x_pos = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(x_pos - width/2, sup_vals, width, label="Supervised", color="#2196F3")
bars2 = ax.bar(x_pos + width/2, raw_vals, width, label="Unsupervised", color="#FF7043")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Valency Validity (%)")
ax.set_title(f"Supervised vs Unsupervised Generation ({n} runs per reaction)")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 115)
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_valency_comparison.png"), dpi=150)
print(f"  -> {OUT}/fig2_valency_comparison.png")
plt.close()


# =====================================================================
# Figure 3: Curriculum penalty ramp
# =====================================================================
from chemistry_constraint_satisfaction.diffusion.training import CurriculumConfig

configs = [
    ("warmup=25%", CurriculumConfig(warmup_fraction=0.25, max_penalty_weight=1.0)),
    ("warmup=50%", CurriculumConfig(warmup_fraction=0.50, max_penalty_weight=1.0)),
    ("no warmup",  CurriculumConfig(warmup_fraction=0.0, max_penalty_weight=1.0)),
]

fig, ax = plt.subplots(figsize=(6, 3.5))
total_epochs = 100
for label, cc in configs:
    scales = [cc.penalty_scale(e, total_epochs) for e in range(total_epochs + 1)]
    ax.plot(range(total_epochs + 1), scales, label=label, linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Penalty weight scale")
ax.set_title("Curriculum Schedule: Penalty Weight Over Training")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_curriculum.png"), dpi=150)
print(f"  -> {OUT}/fig3_curriculum.png")
plt.close()

print("Done.")
