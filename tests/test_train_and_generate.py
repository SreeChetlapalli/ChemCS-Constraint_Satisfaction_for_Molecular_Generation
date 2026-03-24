"""
Integration test: train the diffusion model, then run supervised generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from chemistry_constraint_satisfaction.constraints.chemical_axioms import (
    Atom, MolecularState, check_intermediate,
)
from chemistry_constraint_satisfaction.diffusion.supervisor import Supervisor, GenerationResult
from chemistry_constraint_satisfaction.diffusion.trainer import (
    train_and_export, default_training_molecules,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sn2_reactants():
    ch3br = MolecularState("CH3Br", [
        Atom("C", bonds=4), Atom("Br", bonds=1),
        Atom("H", bonds=1), Atom("H", bonds=1), Atom("H", bonds=1),
    ])
    oh_minus = MolecularState("OH-", [
        Atom("O", bonds=1, formal_charge=-1),
        Atom("H", bonds=1),
    ])
    return [ch3br, oh_minus]


# ===========================================================================
# Tests
# ===========================================================================

class TestDefaultTrainingMolecules:
    def test_returns_non_empty(self):
        mols = default_training_molecules()
        assert len(mols) >= 10

    def test_all_are_molecular_states(self):
        for mol in default_training_molecules():
            assert isinstance(mol, MolecularState)
            assert len(mol.atoms) > 0

    def test_valency_valid(self):
        """All built-in molecules should pass valency checks."""
        for mol in default_training_molecules():
            cr = check_intermediate(mol)
            assert cr.sat, f"{mol.name} failed: {cr.reason}"


class TestTrainAndExport:
    def test_returns_model_and_loss(self):
        npm, loss = train_and_export(
            hidden_dim=16, T=5, epochs=3, steps_per_epoch=10,
            seed=0, verbose=False,
        )
        assert len(loss) == 3
        assert all(isinstance(l, float) for l in loss)

    def test_loss_decreases(self):
        """Over a few epochs the loss should generally decrease."""
        _, loss = train_and_export(
            hidden_dim=16, T=5, epochs=10, steps_per_epoch=20,
            seed=0, verbose=False,
        )
        # First loss should be larger than last (with some tolerance)
        assert loss[-1] < loss[0]


class TestTrainThenGenerate:
    def test_pipeline_completes(self, sn2_reactants):
        """Train, export, and run supervisor — full pipeline smoke test."""
        npm, _ = train_and_export(
            hidden_dim=16, T=5, epochs=5, steps_per_epoch=15,
            seed=42, verbose=False,
        )
        sup = Supervisor(
            npm, sn2_reactants, T=5, max_retries=2,
            max_backtracks=3, verbose=False, prefer_z3=False,
        )
        result = sup.run()
        assert isinstance(result, GenerationResult)
        assert len(result.product.atoms) == sum(
            len(m.atoms) for m in sn2_reactants
        )

    def test_product_valency_valid(self, sn2_reactants):
        """Trained model + supervisor should produce valency-valid products."""
        npm, _ = train_and_export(
            hidden_dim=32, T=10, epochs=10, steps_per_epoch=30,
            seed=42, verbose=False,
        )
        result = Supervisor(
            npm, sn2_reactants, T=10, max_retries=3,
            max_backtracks=5, verbose=False, prefer_z3=False,
        ).run()
        cr = check_intermediate(result.product)
        assert cr.sat, f"Valency violation: {cr.reason}"
