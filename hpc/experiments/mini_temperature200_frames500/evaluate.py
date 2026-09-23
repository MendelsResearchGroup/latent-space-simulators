"""500-frame evaluator extending the frozen physical evaluator with boundary strains."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve()
_ROOT = next(parent for parent in _HERE.parents if (parent / "src/lss").is_dir() and (parent / "notebooks").is_dir())
_PATH = _ROOT / "notebooks/results/all4_coverage_1000/code_v5/evaluate.py"
_SPEC = importlib.util.spec_from_file_location("frozen_all4_evaluate", _PATH)
_BASE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BASE)
_BASE_ROW = _BASE._row


def _widths(graph, sides) -> tuple[float, float]:
    xy = graph.x[:, :2].detach().cpu().numpy()
    return (
        float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean()),
        float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean()),
    )


def _row_with_strain(**kwargs):
    row = _BASE_ROW(**kwargs)
    reference = kwargs["reference"]
    physical_reference = kwargs["physical_reference"]
    sides = kwargs["sides"]
    truth = _BASE.physical(kwargs["truth"], reference)
    prediction = _BASE.physical(kwargs["pred"], reference)
    if kwargs["frame"] == 0:
        truth.x[:, :2] = physical_reference.x[:, :2]
    width0, height0 = _widths(physical_reference, sides)
    true_width, true_height = _widths(truth, sides)
    pred_width, pred_height = _widths(prediction, sides)
    assert np.isfinite([width0, height0, true_width, true_height, pred_width, pred_height]).all()
    row.update({
        "true_boundary_strain_x": (true_width - width0) / width0,
        "true_boundary_strain_y": (true_height - height0) / height0,
        "pred_boundary_strain_x": (pred_width - width0) / width0,
        "pred_boundary_strain_y": (pred_height - height0) / height0,
    })
    return row


_BASE._row = _row_with_strain
evaluate = _BASE.evaluate
physical = _BASE.physical
pref = _BASE._pref
pratio = _BASE._p_ratio
