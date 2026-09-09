"""Post-fit position-derived response evaluation, separate from training.

Both evaluators return per-network rows and source-wise horizon summaries,
including valid/total counts. Reconstruction is the AE ceiling; rollout uses
an autonomous propagator. Estimators and physical-coordinate inversion remain
in the established experiment implementation so historical scores are comparable.
"""
from .experiment import (
    evaluate_autoencoder_reconstruction_horizons,
    evaluate_rollout_horizons,
    rollout_curve_summary,
)

__all__ = [
    "evaluate_autoencoder_reconstruction_horizons",
    "evaluate_rollout_horizons",
    "rollout_curve_summary",
]
