"""Canonical latent-space dynamics API and stable workflow utilities."""

from .canonical import (
    CanonicalDynamicsConfig,
    make_canonical_autoencoder,
    make_canonical_propagator,
)
from .simulator import LatentSpaceSimulator
from .transfer_strain_bundle import DirectionalStrainTransferBundle
from .physics import PhysicsLossConfig, elastic_implicit_euler_energy
from .experiment import (
    initial_latent_analysis,
    prepare_source_spec,
    result_tables,
    rollout_curve_summary,
    save_result_tables,
    seed_everything,
    train_latent_experiment,
)
from .training import (
    LatentNormalizer,
    TrainingConfig,
    TrainingResult,
    train_autoencoder,
    train_propagator,
)

__all__ = [
    "CanonicalDynamicsConfig",
    "LatentNormalizer",
    "LatentSpaceSimulator",
    "DirectionalStrainTransferBundle",
    "PhysicsLossConfig",
    "TrainingConfig",
    "TrainingResult",
    "initial_latent_analysis",
    "make_canonical_autoencoder",
    "make_canonical_propagator",
    "prepare_source_spec",
    "result_tables",
    "rollout_curve_summary",
    "save_result_tables",
    "seed_everything",
    "train_autoencoder",
    "train_latent_experiment",
    "train_propagator",
    "elastic_implicit_euler_energy",
]
