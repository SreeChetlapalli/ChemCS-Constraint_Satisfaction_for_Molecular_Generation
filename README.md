# Chemistry Constraint Satisfaction

**Constraint-checked molecular generation** — Z3 (or a pure-Python fallback) verifies mass/charge conservation and bond valency while a small diffusion-style model runs inside a **supervisor** loop.

---

## Navigate this repo

| I want to… | Go here |
|------------|---------|
| **Run something in 2 minutes** | [Quick start](#quick-start) |
| **Install and import the package** | [Setup](#setup) · [docs/USAGE.md](docs/USAGE.md) |
| **Run tests** | [Tests](#tests) · [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Try a full demo (CLI)** | `python scripts/demo.py` |
| **Try Jupyter / Colab** | [notebooks/README.md](notebooks/README.md) |
| **See every folder** | [Repository map](#repository-map) |
| **Change the code** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Doc index** | [docs/README.md](docs/README.md) |

---

## Table of contents

- [Navigate this repo](#navigate-this-repo)
- [Quick start](#quick-start)
- [What this project does](#what-this-project-does)
- [Repository map](#repository-map)
- [Setup](#setup)
- [Tests](#tests)
- [Scripts](#scripts)
- [Design notes & trade-offs](#design-notes--trade-offs)
- [Goals (example timeline)](#goals-example-timeline)
- [Contributing & license](#contributing--license)

---

## Quick start

1. **Clone** the repository and `cd` into the folder (name may be `ChemistryConstraintSatisfaction` or `ChemistryConstraintSatisfaction-1` depending on how you cloned).

2. **Create a venv** (Python **3.10+** recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install** (pick one):

   ```bash
   # Recommended: editable install so `import chemistry_constraint_satisfaction` works everywhere
   pip install -e ".[dev]"
   ```

   ```bash
   # Alternative: requirements file only (you may need to set PYTHONPATH=src for some tools)
   pip install -r requirements.txt
   ```

4. **Verify**:

   ```bash
   python scripts/check_env.py
   ```

   You want to see PyTorch, Z3, and the package version, ending with `OK — environment ready.`

5. **Run tests**:

   ```bash
   python run_tests.py
   ```

6. **Run the demo**:

   ```bash
   python scripts/demo.py
   ```

For copy-paste **Python examples**, see **[docs/USAGE.md](docs/USAGE.md)**.

---

## What this project does

Generative models can propose **invalid** structures (e.g. wrong valency, broken conservation). This repo wraps a small **NumPy** graph denoising model with a **supervisor** that:

1. Decodes each step to a `MolecularState`.
2. Runs **constraint checks** (valency during the trajectory; full reaction check at the end).
3. **Corrects or backtracks** within configured limits (`max_retries`, `max_backtracks`).

If `z3-solver` is installed, checks can use Z3; otherwise a **pure-Python** checker is used.

---

## Repository map

```
ChemistryConstraintSatisfaction/
├── README.md                 ← You are here (overview + navigation)
├── CONTRIBUTING.md           ← Tests, layout, how to contribute
├── pyproject.toml            ← Package metadata + editable install + pytest config
├── requirements.txt          ← Same runtime deps as pyproject (pip -r friendly)
├── run_tests.py              ← Run all tests with stdlib unittest only
│
├── docs/
│   ├── README.md             ← Index of extra documentation
│   └── USAGE.md              ← Import examples, module map
│
├── src/chemistry_constraint_satisfaction/   ← Installable Python package
│   ├── __init__.py           ← Package version
│   ├── constraints/          ← Atoms, molecules, check_reaction / check_intermediate
│   └── diffusion/            ← MolecularDiffusionModel, Supervisor, encode/decode
│
├── tests/                    ← pytest suites (also mirrored in run_tests.py)
├── scripts/
│   ├── check_env.py          ← Quick environment sanity check
│   └── demo.py               ← End-to-end CLI demo + small benchmark
└── notebooks/
    ├── README.md             ← How to open demo.ipynb (local / Colab)
    └── demo.ipynb            ← Interactive walkthrough + optional PyTorch training
```

---

## Setup

### Clone

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### Dependencies

- **Required:** `numpy`, `torch`, `z3-solver`, `tqdm` (see `requirements.txt` or `pyproject.toml`).
- **Optional:** RDKit (often via Conda) for SMILES / extra chemistry tooling — not required for the core demos.

```bash
conda install -c conda-forge rdkit
```

### GPU (optional)

- Local: install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org).
- Colab: use `notebooks/demo.ipynb` and set the runtime to GPU.

---

## Tests

| Command | When to use |
|---------|-------------|
| `python run_tests.py` | No pytest installed; uses **unittest** only |
| `pytest tests/ -v` | After `pip install -e ".[dev]"` |

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for details.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `python scripts/check_env.py` | Confirms PyTorch, Z3, and package import |
| `python scripts/demo.py` | Constraint demos + supervised generation + short benchmark |

---

## Design notes & trade-offs

- Steps are only committed when they pass the configured checks (or after correction).
- Z3 adds **runtime cost**; use `prefer_z3=False` or the pure-Python path when you need speed over solver-backed checks.
- Final validity still depends on the **model**; the supervisor enforces **checked** constraints, not “magic chemistry.”

---

## Goals (example timeline)

| Milestone | Target |
|-----------|--------|
| Mid-term | Step-level correction with Z3-backed checks |
| End-term | Larger benchmark vs a baseline generator |

---

## Contributing & license

- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **License:** Use and cite as appropriate for your institution. This project builds on open-source tools (e.g. PyTorch, Z3).

---

## Publishing note

If you fork this repo, update **`pyproject.toml`** `name` / `version` as needed before publishing to PyPI. The `[project.urls]` block is intentionally omitted so you can add your real repository URL when you publish.
