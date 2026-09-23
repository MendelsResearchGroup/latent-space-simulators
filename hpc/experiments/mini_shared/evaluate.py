"""Dense physical-coordinate validation evaluation for the Reid latent run.

This module is deliberately a post-fit diagnostic.  It does not train, select
checkpoints, or use any response-derived quantity as a model input.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from graph_utils.box import Box
from thermal_response import apply_mixed_fits
from lss.data import POSITION_NORMALIZATION
from lss.graph import clone_graph
from lss.dynamics.training import (
    decode_latent_to_graph,
    encode_frame_latent,
    encode_reference_context,
    latent_step,
)


def physical(graph, reference):
    """Invert saved position normalization with the frame-zero reference box."""
    if str(getattr(reference, "coordinate_normalization", "")).lower() != POSITION_NORMALIZATION:
        raise ValueError("Expected a position-normalized frame-zero reference graph.")
    out = clone_graph(graph).cpu()
    half = reference.reference_box_half_extent.detach().cpu().double()[:2]
    center = reference.reference_box_center.detach().cpu().double()[:2]
    out.x = out.x.detach().cpu().double().clone()
    out.x[:, :2] = out.x[:, :2] * half + center
    box = out.box
    out.box = Box(
        float(box.x1 * half[0] + center[0]), float(box.x2 * half[0] + center[0]),
        float(box.y1 * half[1] + center[1]), float(box.y2 * half[1] + center[1]),
        float(getattr(box, "z1", -0.1)), float(getattr(box, "z2", 0.1)),
    )
    out.box_tensor = half * torch.as_tensor(graph.box_tensor, dtype=torch.float64)[:2]
    out.coordinate_normalization = "physical"
    return out


def _component_stats(truth: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    """SSE and target moments for componentwise-centered, variance-weighted R²."""
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("Expected matching (nodes, 2) physical-coordinate arrays.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        return {
            "sse": float("nan"), "target_sum_x": float("nan"), "target_sum_y": float("nan"),
            "target_sumsq_x": float("nan"), "target_sumsq_y": float("nan"),
            "target_count_per_component": float("nan"), "target_component_count": float("nan"),
            "sst": float("nan"), "r2": float("nan"), "valid": False,
        }
    count = int(truth.shape[0])
    sums = truth.sum(axis=0)
    sumsq = (truth * truth).sum(axis=0)
    sse = float(((pred - truth) ** 2).sum())
    sst = float((sumsq - sums * sums / count).sum()) if count else float("nan")
    return {
        "sse": sse,
        "target_sum_x": float(sums[0]), "target_sum_y": float(sums[1]),
        "target_sumsq_x": float(sumsq[0]), "target_sumsq_y": float(sumsq[1]),
        "target_count_per_component": count,
        "target_component_count": 2 * count,
        "sst": sst,
        "r2": _r2(sse, sst),
        "valid": True,
    }


def _r2(sse: float, sst: float) -> float:
    return float(1.0 - sse / sst) if math.isfinite(sst) and sst > 0.0 else float("nan")


def _pref(reference):
    pref = physical(reference, reference)
    error = float((pref.x[:, :2] - reference.reference_context_positions.cpu().double()[:, :2]).abs().max())
    if error > 2e-5:
        raise ValueError(f"Physical reference inversion mismatch: {error}.")
    # Preserve exact source coordinates before directional quantile tie-breaking.
    pref.x[:, :2] = reference.reference_context_positions.cpu().double()[:, :2]
    return pref, error


def _p_ratio(reference, position_graph, sides) -> float:
    return float(calc_p_ratio_rollout_sides([reference, position_graph], -1, side_idx=sides))


def _row(*, mode: str, split: str, seed: int, latent_dim: int, validation_index: int,
         source_validation_index: int, source: str, frame: int, reference,
         physical_reference, sides, truth, pred, previous_truth, previous_pred) -> dict:
    truth_phys, pred_phys = physical(truth, reference), physical(pred, reference)
    if frame == 0:
        # The normalizer inverse is accurate but not bit-identical to the
        # frozen physical source coordinates.  Frame zero is exactly reference.
        truth_phys.x[:, :2] = physical_reference.x[:, :2]
    truth_xy = truth_phys.x[:, :2].numpy()
    pred_xy = pred_phys.x[:, :2].numpy()
    ref_xy = physical_reference.x[:, :2].numpy()
    displacement = _component_stats(truth_xy - ref_xy, pred_xy - ref_xy)
    zero = _component_stats(truth_xy - ref_xy, np.zeros_like(truth_xy))
    row = {
        "mode": mode, "split": split, "seed": int(seed), "latent_dim": int(latent_dim),
        "validation_index": int(validation_index),
        "source_validation_index": int(source_validation_index), "source": source,
        "frame": int(frame), "nodes": int(truth_xy.shape[0]),
        "true_p_ratio": _p_ratio(physical_reference, truth_phys, sides),
        "pred_p_ratio": _p_ratio(physical_reference, pred_phys, sides),
        "zero_p_ratio": _p_ratio(physical_reference, physical_reference, sides),
        "p_ratio_method": "physical_endpoint_fixed_frame0_sides_quantile_0.10",
    }
    for kind, xy in (("true", truth_xy), ("pred", pred_xy)):
        row[kind+"_width"] = float(xy[sides["right"], 0].mean()-xy[sides["left"], 0].mean())
        row[kind+"_height"] = float(xy[sides["top"], 1].mean()-xy[sides["bottom"], 1].mean())
    row.update({f"displacement_{key}": value for key, value in displacement.items()})
    row.update({f"zero_displacement_{key}": value for key, value in zero.items()})
    row["displacement_total"] = 1
    row["displacement_valid"] = int(displacement["valid"])
    row["zero_displacement_total"] = 1
    row["zero_displacement_valid"] = int(zero["valid"])
    row["p_ratio_valid"] = bool(math.isfinite(row["true_p_ratio"]) and math.isfinite(row["pred_p_ratio"]))
    row["zero_p_ratio_valid"] = bool(math.isfinite(row["true_p_ratio"]) and math.isfinite(row["zero_p_ratio"]))
    row["p_ratio_total"] = 1
    row["zero_p_ratio_total"] = 1
    if frame == 0:
        for key in displacement:
            row[f"increment_{key}"] = float("nan")
            row[f"zero_increment_{key}"] = float("nan")
        row["increment_total"] = 0
        row["increment_valid"] = 0
        row["zero_increment_total"] = 0
        row["zero_increment_valid"] = 0
        return row
    previous_truth_xy = physical(previous_truth, reference).x[:, :2].numpy()
    previous_pred_xy = physical(previous_pred, reference).x[:, :2].numpy()
    if frame == 1:
        previous_truth_xy = ref_xy
        if mode == 'autonomous_delta_mlp':
            previous_pred_xy = ref_xy
    increment = _component_stats(truth_xy - previous_truth_xy, pred_xy - previous_pred_xy)
    row.update({f"increment_{key}": value for key, value in increment.items()})
    zero_increment = _component_stats(truth_xy - previous_truth_xy, np.zeros_like(truth_xy))
    row.update({f"zero_increment_{key}": value for key, value in zero_increment.items()})
    row["increment_total"] = 1
    row["increment_valid"] = int(increment["valid"])
    row["zero_increment_total"] = 1
    row["zero_increment_valid"] = int(zero_increment["valid"])
    return row


def _pooled_r2(group: pd.DataFrame, prefix: str) -> float:
    valid = group.loc[group[f"{prefix}_valid"].eq(1)]
    if valid.empty:
        return float("nan")
    count = valid[f"{prefix}_target_count_per_component"].sum(min_count=1)
    if not math.isfinite(count) or count <= 0:
        return float("nan")
    sums = valid[[f"{prefix}_target_sum_x", f"{prefix}_target_sum_y"]].sum(min_count=1).to_numpy(float)
    sumsq = valid[[f"{prefix}_target_sumsq_x", f"{prefix}_target_sumsq_y"]].sum(min_count=1).to_numpy(float)
    sse = float(valid[f"{prefix}_sse"].sum(min_count=1))
    sst = float((sumsq - sums * sums / count).sum())
    return _r2(sse, sst)


def _summaries(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (mode, split, source, frame), group in rows.groupby(["mode", "split", "source", "frame"], sort=True):
        record = {
            "mode": mode, "split": split, "source": source, "frame": int(frame), "networks": int(len(group)),
            "nodes": int(group["nodes"].sum()),
            "p_ratio_valid": int(group["p_ratio_valid"].sum()),
            "p_ratio_total": int(group["p_ratio_total"].sum()),
            "p_ratio_r2": _scalar_r2(group, "true_p_ratio", "pred_p_ratio"),
            "zero_p_ratio_r2": _scalar_r2(group, "true_p_ratio", "zero_p_ratio"),
            "zero_p_ratio_valid": int(group["zero_p_ratio_valid"].sum()),
            "zero_p_ratio_total": int(group["zero_p_ratio_total"].sum()),
        }
        for prefix in ("displacement", "zero_displacement", "increment", "zero_increment"):
            record[f"{prefix}_pooled_r2"] = _pooled_r2(group, prefix)
            valid = group.loc[group[f"{prefix}_valid"].eq(1)]
            record[f"{prefix}_valid"] = int(group[f"{prefix}_valid"].sum())
            record[f"{prefix}_total"] = int(group[f"{prefix}_total"].sum())
            record[f"{prefix}_sse"] = float(valid[f"{prefix}_sse"].sum(min_count=1))
            record[f"{prefix}_target_component_count"] = float(valid[f"{prefix}_target_component_count"].sum(min_count=1))
        records.append(record)
    return pd.DataFrame(records)


def _scalar_r2(group: pd.DataFrame, truth: str, pred: str) -> float:
    finite = group[[truth, pred]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2:
        return float("nan")
    y = finite[truth].to_numpy(float)
    denominator = float(((y - y.mean()) ** 2).sum())
    return _r2(float(((finite[pred].to_numpy(float) - y) ** 2).sum()), denominator)


def _evaluate(result: dict, *, seed: int, latent_dim: int, output: Path,
              horizon: int, include_autonomous: bool, split: str) -> pd.DataFrame:
    """Evaluate all validation trajectories at frames 0 through ``horizon``.

    Reconstruction encodes each observed target frame.  Autonomous rollout
    encodes frame zero once and subsequently advances only its own latent state.
    Both decoded graphs inherit topology and static metadata only from frame zero.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    required = ("ae", "normalizers", "params")
    if include_autonomous:
        required += ("dyn", "latent_stats")
    if any(key not in result for key in required):
        raise KeyError(f"Result lacks required evaluation keys: {required}.")
    ae, normalizers, params = (result[key] for key in ("ae", "normalizers", "params"))
    ae.eval()
    if include_autonomous:
        dyn, stats = (result[key] for key in ("dyn", "latent_stats"))
        dyn.eval()
    device = next(ae.parameters()).device
    rows = []
    source_counts: dict[str, int] = {}
    with torch.no_grad():
        for validation_index, sim in enumerate(result["val_data"]):
            if len(sim) <= horizon:
                raise ValueError(f"Validation simulation {validation_index} has {len(sim)} frames; need {horizon + 1}.")
            reference = sim[0]
            source = str(reference.source_name)
            trajectory_id = int(reference.source_sim_id)
            source_validation_index = source_counts.get(source, 0)
            source_counts[source] = source_validation_index + 1
            physical_reference, _ = _pref(reference)
            sides = directional_side_indices_from_box(physical_reference, quantile=0.10)
            reconstructions = []
            for frame in range(horizon + 1):
                z = encode_frame_latent(ae, sim, frame, pos_dim=2,
                    node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device=device)
                reconstructions.append(decode_latent_to_graph(ae, sim, z, frame, pos_dim=2,
                    ae_target_mode=params["ae_target_mode"], normalizers=normalizers, device=device))
            modes = [("ae_reconstruction", reconstructions)]
            if include_autonomous:
                rollout = [clone_graph(reference).cpu()]
                z = encode_frame_latent(ae, sim, 0, pos_dim=2,
                    node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device=device)
                context = encode_reference_context(ae, sim, pos_dim=2, normalizers=normalizers,
                    device=device, include_temperature=False, include_source_id=False, pool_mode="mean")
                for frame in range(1, horizon + 1):
                    z = latent_step(dyn, z, stats, context=context, loss_mode="delta")
                    rollout.append(decode_latent_to_graph(ae, sim, z, frame, pos_dim=2,
                        ae_target_mode=params["ae_target_mode"], normalizers=normalizers, device=device))
                modes.append(("autonomous_delta_mlp", rollout))
            for mode, predictions in modes:
                for frame, prediction in enumerate(predictions):
                    row = _row(mode=mode, split=split, seed=seed, latent_dim=latent_dim,
                        validation_index=validation_index, source_validation_index=source_validation_index,
                        source=source, frame=frame, reference=reference,
                        physical_reference=physical_reference, sides=sides, truth=sim[frame], pred=prediction,
                        previous_truth=None if frame == 0 else sim[frame - 1],
                        previous_pred=None if frame == 0 else predictions[frame - 1])
                    row["trajectory_id"] = trajectory_id
                    rows.append(row)
    table = apply_mixed_fits(pd.DataFrame(rows))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "per_network.csv", index=False)
    _summaries(table).to_csv(output / "per_step.csv", index=False)
    return table


def evaluate(result: dict, *, seed: int, latent_dim: int, output: Path,
             horizon: int = 100, split: str = "validation") -> pd.DataFrame:
    """Evaluate frozen AE reconstruction and autonomous delta-MLP rollout."""
    return _evaluate(result, seed=seed, latent_dim=latent_dim, output=output,
                     horizon=horizon, include_autonomous=True, split=split)


def evaluate_autoencoder(result: dict, *, seed: int, latent_dim: int, output: Path,
                         horizon: int = 100, split: str = "validation") -> pd.DataFrame:
    """Evaluate a frozen AE without requiring a propagator in ``result``."""
    return _evaluate(result, seed=seed, latent_dim=latent_dim, output=output,
                     horizon=horizon, include_autonomous=False, split=split)


def _metric_self_test() -> None:
    truth = np.array([[1.0, 3.0], [3.0, 7.0]])
    perfect = _component_stats(truth, truth)
    assert perfect["r2"] == 1.0 and perfect["target_component_count"] == 4
    shifted = _component_stats(truth, truth + 1.0)
    assert np.isclose(shifted["sst"], 10.0) and np.isclose(shifted["r2"], 0.6)
    constant = _component_stats(np.ones((2, 2)), np.zeros((2, 2)))
    assert math.isnan(constant["r2"])


if __name__ == "__main__":
    _metric_self_test()
