# Contributing

Thanks for helping improve this project.

## Environment

1. Clone the repo and create a virtual environment (see [README.md](README.md#setup)).
2. Install in editable mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

   Or use `pip install -r requirements.txt` and install `pytest` separately if you prefer.

## Running tests

**Option A — no pytest required (stdlib only):**

```bash
python run_tests.py
```

**Option B — pytest (from repo root):**

```bash
pytest tests/ -v
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest so imports resolve without manual `PYTHONPATH`.

## Code layout (where to edit)

| Path | What it contains |
|------|------------------|
| `src/chemistry_constraint_satisfaction/constraints/` | Chemical checks: `chemical_axioms.py` |
| `src/chemistry_constraint_satisfaction/diffusion/` | NumPy diffusion model + `supervisor.py` |
| `tests/` | Pytest suites mirroring the above |
| `run_tests.py` | Duplicate coverage with `unittest` (CI-friendly without pytest) |
| `scripts/` | `demo.py`, `check_env.py` |
| `notebooks/` | Colab/Jupyter walkthrough |

## Style

- Prefer clear names and short docstrings on public APIs.
- Keep behavior changes covered by tests when possible.

## Pull requests

- Describe what changed and how you tested it (`python run_tests.py` and/or `pytest`).
