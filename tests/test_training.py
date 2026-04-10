"""Tests for the PyTorch training module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np

torch = pytest.importorskip("torch", reason="PyTorch required for training tests")

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState,
)
from chemistry_constraint_satisfaction.diffusion.training import (
    TorchMolecularDenoiser,
    GraphConvLayer,
    cosine_alpha_bar,
    linear_alpha_bar,
    compute_loss,
    LossWeights,
    CurriculumConfig,
    train,
    TrainConfig,
    export_to_numpy,
    molecules_to_tensors,
)
from chemistry_constraint_satisfaction.diffusion.model import ATOM_FEAT_DIM


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def water():
    return MolecularState("H2O", [
        Atom("O", bonds=2), Atom("H", bonds=1), Atom("H", bonds=1),
    ])


@pytest.fixture
def methane():
    return MolecularState("CH4", [
        Atom("C", bonds=4),
        Atom("H", bonds=1), Atom("H", bonds=1),
        Atom("H", bonds=1), Atom("H", bonds=1),
    ])


@pytest.fixture
def denoiser():
    return TorchMolecularDenoiser(hidden_dim=16, num_layers=2)


# ===========================================================================
# Tests: TorchMolecularDenoiser
# ===========================================================================

class TestDenoiser:
    def test_output_shapes_unbatched(self, denoiser):
        N = 4
        x = torch.randn(N, ATOM_FEAT_DIM)
        adj = torch.rand(N, N)
        t = torch.tensor(0.5)
        x0, bond_logits = denoiser(x, adj, t)
        assert x0.shape == (N, ATOM_FEAT_DIM)
        assert bond_logits.shape == (N, N, 4)

    def test_output_shapes_batched(self, denoiser):
        B, N = 3, 4
        x = torch.randn(B, N, ATOM_FEAT_DIM)
        adj = torch.rand(B, N, N)
        t = torch.rand(B)
        x0, bond_logits = denoiser(x, adj, t)
        assert x0.shape == (B, N, ATOM_FEAT_DIM)
        assert bond_logits.shape == (B, N, N, 4)

    def test_gradients_flow(self, denoiser):
        N = 3
        x = torch.randn(N, ATOM_FEAT_DIM, requires_grad=True)
        adj = torch.rand(N, N)
        t = torch.tensor(0.5)
        x0, bl = denoiser(x, adj, t)
        loss = x0.sum() + bl.sum()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in denoiser.parameters())


class TestGraphConvLayer:
    def test_output_shape(self):
        layer = GraphConvLayer(ATOM_FEAT_DIM, 32)
        x = torch.randn(5, ATOM_FEAT_DIM)
        adj = torch.eye(5)
        out = layer(x, adj)
        assert out.shape == (5, 32)

    def test_relu_applied(self):
        layer = GraphConvLayer(ATOM_FEAT_DIM, 32)
        x = torch.randn(5, ATOM_FEAT_DIM)
        adj = torch.eye(5)
        out = layer(x, adj)
        assert (out >= 0).all()


# ===========================================================================
# Tests: Noise schedules
# ===========================================================================

class TestNoiseSchedules:
    def test_cosine_at_zero(self):
        ab = cosine_alpha_bar(torch.tensor(0.0))
        assert ab.item() == pytest.approx(1.0, abs=0.01)

    def test_cosine_at_one(self):
        ab = cosine_alpha_bar(torch.tensor(1.0))
        assert ab.item() < 0.05

    def test_cosine_monotonic(self):
        t = torch.linspace(0, 1, 50)
        ab = cosine_alpha_bar(t)
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-5).all()

    def test_linear_at_zero(self):
        ab = linear_alpha_bar(torch.tensor(0.0))
        assert ab.item() == pytest.approx(1.0, abs=0.01)

    def test_linear_monotonic(self):
        t = torch.linspace(0, 1, 50)
        ab = linear_alpha_bar(t)
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-5).all()


# ===========================================================================
# Tests: Loss computation
# ===========================================================================

class TestLoss:
    def test_loss_positive(self):
        N = 4
        x0_pred = torch.randn(N, ATOM_FEAT_DIM)
        bond_logits = torch.randn(N, N, 4)
        x_clean = torch.randn(N, ATOM_FEAT_DIM)
        adj_clean = torch.zeros(N, N)
        loss, breakdown = compute_loss(x0_pred, bond_logits, x_clean, adj_clean)
        assert loss.item() > 0
        assert "total" in breakdown

    def test_loss_all_components_present(self):
        N = 3
        x0_pred = torch.randn(N, ATOM_FEAT_DIM)
        bond_logits = torch.randn(N, N, 4)
        x_clean = torch.randn(N, ATOM_FEAT_DIM)
        adj_clean = torch.zeros(N, N)
        _, breakdown = compute_loss(x0_pred, bond_logits, x_clean, adj_clean)
        for key in ["denoising", "bond_ce", "valency_penalty", "element_conservation"]:
            assert key in breakdown

    def test_custom_weights(self):
        N = 3
        x0_pred = torch.randn(N, ATOM_FEAT_DIM)
        bond_logits = torch.randn(N, N, 4)
        x_clean = torch.randn(N, ATOM_FEAT_DIM)
        adj_clean = torch.zeros(N, N)
        w = LossWeights(denoising=10.0, bond_ce=0.0, valency_penalty=0.0, element_conservation=0.0)
        loss_high, _ = compute_loss(x0_pred, bond_logits, x_clean, adj_clean, w)
        w2 = LossWeights(denoising=0.1, bond_ce=0.0, valency_penalty=0.0, element_conservation=0.0)
        loss_low, _ = compute_loss(x0_pred, bond_logits, x_clean, adj_clean, w2)
        assert loss_high.item() > loss_low.item()


# ===========================================================================
# Tests: Curriculum
# ===========================================================================

class TestCurriculum:
    def test_warmup_returns_zero(self):
        c = CurriculumConfig(warmup_fraction=0.5, max_penalty_weight=1.0)
        assert c.penalty_scale(0, 100) == 0.0
        assert c.penalty_scale(25, 100) == 0.0

    def test_after_warmup_grows(self):
        c = CurriculumConfig(warmup_fraction=0.25, max_penalty_weight=1.0)
        assert c.penalty_scale(25, 100) == 0.0
        mid = c.penalty_scale(62, 100)
        end = c.penalty_scale(100, 100)
        assert mid > 0
        assert end >= mid

    def test_max_reached(self):
        c = CurriculumConfig(warmup_fraction=0.0, max_penalty_weight=2.0)
        assert c.penalty_scale(100, 100) == pytest.approx(2.0)


# ===========================================================================
# Tests: Training loop (short smoke test)
# ===========================================================================

class TestTrainLoop:
    def test_short_train(self, water, methane):
        cfg = TrainConfig(
            lr=1e-3, epochs=5, hidden_dim=16, num_layers=2,
            log_every=100, device="cpu",
        )
        result = train([water, methane], config=cfg, verbose=False)
        assert len(result.epoch_losses) == 5
        assert result.best_loss < float("inf")

    def test_loss_decreases(self, water, methane):
        cfg = TrainConfig(
            lr=1e-3, epochs=30, hidden_dim=16, num_layers=2,
            log_every=100, device="cpu",
        )
        result = train([water, methane], config=cfg, verbose=False)
        first_5 = np.mean([d["total"] for d in result.epoch_losses[:5]])
        last_5 = np.mean([d["total"] for d in result.epoch_losses[-5:]])
        assert last_5 < first_5

    def test_empty_molecules_raises(self):
        with pytest.raises(ValueError):
            train([], verbose=False)


# ===========================================================================
# Tests: Export to NumPy
# ===========================================================================

class TestExport:
    def test_export_produces_numpy_model(self, water, methane):
        cfg = TrainConfig(lr=1e-3, epochs=3, hidden_dim=16, log_every=100, device="cpu")
        result = train([water, methane], config=cfg, verbose=False)
        np_model = export_to_numpy(result.final_model)
        from chemistry_constraint_satisfaction.diffusion.model import MolecularDiffusionModel
        assert isinstance(np_model, MolecularDiffusionModel)
        assert np_model.hidden_dim == 16

    def test_export_can_run_inference(self, water, methane):
        cfg = TrainConfig(lr=1e-3, epochs=3, hidden_dim=16, log_every=100, device="cpu")
        result = train([water, methane], config=cfg, verbose=False)
        np_model = export_to_numpy(result.final_model)
        from chemistry_constraint_satisfaction.diffusion.model import encode_molecule
        x, adj = encode_molecule(water)
        x_n, adj_n = np_model.forward_noisy(x, adj, t=5, T=10)
        x_prev, adj_prev = np_model.reverse_step(x_n, adj_n, t=5, T=10)
        assert x_prev.shape == x.shape


# ===========================================================================
# Tests: Dataset helpers
# ===========================================================================

class TestDataHelpers:
    def test_molecules_to_tensors(self, water, methane):
        pairs = molecules_to_tensors([water, methane])
        assert len(pairs) == 2
        x_w, adj_w = pairs[0]
        assert isinstance(x_w, torch.Tensor)
        assert x_w.shape[0] == len(water.atoms)
