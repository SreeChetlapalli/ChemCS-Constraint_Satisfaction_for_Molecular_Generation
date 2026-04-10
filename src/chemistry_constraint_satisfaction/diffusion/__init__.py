"""Diffusion model + supervisor."""

from .model import (
    MolecularDiffusionModel,
    encode_molecule,
    atom_to_feat,
    feat_to_atom,
    ATOM_FEAT_DIM,
    NUM_ELEM,
    ELEMENTS,
)
from .supervisor import Supervisor, GenerationResult, StepRecord

__all__ = [
    "MolecularDiffusionModel",
    "encode_molecule",
    "atom_to_feat",
    "feat_to_atom",
    "ATOM_FEAT_DIM",
    "NUM_ELEM",
    "ELEMENTS",
    "Supervisor",
    "GenerationResult",
    "StepRecord",
]

def _lazy_training():
    """Import training stuff on demand (avoids torch import at load time)."""
    from .training import (
        TorchMolecularDenoiser,
        train,
        TrainConfig,
        TrainResult,
        export_to_numpy,
        CurriculumConfig,
        LossWeights,
    )
    return {
        "TorchMolecularDenoiser": TorchMolecularDenoiser,
        "train": train,
        "TrainConfig": TrainConfig,
        "TrainResult": TrainResult,
        "export_to_numpy": export_to_numpy,
        "CurriculumConfig": CurriculumConfig,
        "LossWeights": LossWeights,
    }
