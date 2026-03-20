# Notebooks

## `demo.ipynb`

End-to-end walkthrough:

1. Install dependencies (Z3, etc.) — Colab-friendly cells at the top.
2. Constraint checking on example molecules and reactions.
3. One supervised diffusion run (`Supervisor`).
4. Optional PyTorch training loop for the denoising GNN and simple benchmarks.

### Open locally

```bash
# from repo root, with venv activated
pip install -e ".[dev]"   # or pip install -r requirements.txt + jupyter
jupyter notebook notebooks/demo.ipynb
```

### Open in Google Colab

Upload the repo or clone it in a Colab cell, then open `notebooks/demo.ipynb`. Edit the clone URL in the setup cell to match your fork. Enable **Runtime → Change runtime type → GPU** if you want faster training.
