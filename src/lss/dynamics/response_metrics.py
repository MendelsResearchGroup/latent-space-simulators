"""Finite-mask metrics for scalar physical-response predictions."""
from __future__ import annotations

import numpy as np


def scalar_prediction_metrics(y_true, y_pred) -> dict[str, float | int]:
    """Score scalar predictions with a shared finite-pair mask.

    The R² is target-centered: it compares squared residuals with variation in
    the finite targets of the same valid pairs.  It is undefined for constant
    targets, including a single valid pair.
    """
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must contain the same number of values.")

    true_finite = np.isfinite(true)
    pred_finite = np.isfinite(pred)
    valid_mask = true_finite & pred_finite
    valid_true = true[valid_mask]
    valid_pred = pred[valid_mask]
    denominator = float(np.square(valid_true - valid_true.mean()).sum()) if len(valid_true) else 0.0

    return {
        "p_ratio_r2": (
            float(1.0 - np.square(valid_true - valid_pred).sum() / denominator)
            if denominator > 0.0
            else float("nan")
        ),
        "mae": float(np.abs(valid_true - valid_pred).mean()) if len(valid_true) else float("nan"),
        "target_variance": float(np.var(valid_true)) if len(valid_true) else float("nan"),
        "total": int(len(true)),
        "valid": int(valid_mask.sum()),
        "true_valid": int(true_finite.sum()),
        "true_nonfinite": int((~true_finite).sum()),
        "pred_nonfinite": int((~pred_finite).sum()),
    }


__all__ = ["scalar_prediction_metrics"]
