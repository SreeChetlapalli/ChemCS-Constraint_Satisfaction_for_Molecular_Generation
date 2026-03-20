# Usage guide

How to use this project from Python after [installing](../README.md#setup).

## Install the package (recommended)

From the repository root:

```bash
pip install -e .
```

With test tools:

```bash
pip install -e ".[dev]"
```

This registers `chemistry_constraint_satisfaction` on your Python path so imports work from any working directory.

## Minimal examples

### Check a reaction (mass, charge, valency)

```python
from chemistry_constraint_satisfaction.constraints import (
    Atom,
    MolecularState,
    check_reaction,
)

reactants = [
    MolecularState("CH3Br", [
        Atom("C", 4), Atom("Br", 1),
        Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("OH-", [
        Atom("O", 1, formal_charge=-1),
        Atom("H", 1),
    ]),
]
products = [
    MolecularState("CH3OH", [
        Atom("C", 4), Atom("O", 2),
        Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1),
    ]),
    MolecularState("Br-", [Atom("Br", 0, formal_charge=-1)]),
]

result = check_reaction(reactants, products)
print(result.sat, result.reason)
```

### Run supervised generation (diffusion + supervisor)

```python
from chemistry_constraint_satisfaction.diffusion import (
    MolecularDiffusionModel,
    Supervisor,
)

model = MolecularDiffusionModel(hidden_dim=64, seed=42)
sup = Supervisor(
    model,
    reactants=reactants,
    T=20,
    verbose=True,
)
out = sup.run()
print(out.success, len(out.product.atoms))
```

## Where the code lives

| Area | Module path | Role |
|------|-------------|------|
| Atoms / molecules / checks | `chemistry_constraint_satisfaction.constraints` | `Atom`, `MolecularState`, `check_reaction`, `check_intermediate` |
| Diffusion model | `chemistry_constraint_satisfaction.diffusion.model` | `MolecularDiffusionModel`, `encode_molecule` |
| Supervisor loop | `chemistry_constraint_satisfaction.diffusion.supervisor` | `Supervisor`, `GenerationResult` |

## Z3 vs pure Python

- If `z3-solver` is installed, `check_reaction(..., prefer_z3=True)` can use the solver.
- Otherwise the same API falls back to a pure-Python checker (`Z3_AVAILABLE` in `constraints`).

## Scripts and notebooks

- **CLI-style demo:** `python scripts/demo.py` (constraint checks + one supervised run + small benchmark).
- **Interactive / training:** open `notebooks/demo.ipynb` in Jupyter or Google Colab (see `notebooks/README.md`).

## Further reading

- [Contributing](../CONTRIBUTING.md) — tests and how to change the code.
- [Repository map](../README.md#repository-map) — every folder explained.
