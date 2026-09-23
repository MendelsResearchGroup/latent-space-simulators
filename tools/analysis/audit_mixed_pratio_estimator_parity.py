"""Audit mixed-temperature p-ratio parity without model inference.

This audit reconstructs the successful 06b five-seed endpoint score from its
saved per-network rows and independently recomputes notebook-06 endpoint
p-ratios from cached physical side widths/heights using the historical
minimum-reciprocal side-displacement definition.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = ROOT / "notebooks/results/06b_4d_reconstruction"
MINI_ROWS = ROOT / "notebooks/results/mini_shared/thermal_fit/joint_original/test/per_network.csv"
OUT = ROOT / "notebooks/results/mini_shared/mixed_pratio_estimator_parity_20260913.json"
SEEDS = (3456456, 123, 456, 786, 2026)
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(truth: np.ndarray, pred: np.ndarray) -> float:
    finite = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[finite], pred[finite]
    denominator = float(np.square(truth - truth.mean()).sum()) if len(truth) else 0.0
    return float(1.0 - np.square(truth - pred).sum() / denominator) if denominator > 0 else float("nan")


def historical_endpoint(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    out = np.full(np.broadcast_shapes(dx.shape, dy.shape), np.nan, dtype=float)
    finite = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > EPS) & (np.abs(dy) > EPS)
    p1 = np.full_like(out, np.nan)
    p2 = np.full_like(out, np.nan)
    p1[finite] = -dy[finite] / dx[finite]
    p2[finite] = -dx[finite] / dy[finite]
    choose_p1 = np.abs(p1) < np.abs(p2)
    out[finite] = np.where(choose_p1[finite], p1[finite], p2[finite])
    return out


def old_score_reconstruction() -> dict:
    per_seed = []
    inputs = []
    for seed in SEEDS:
        path = OLD_ROOT / f"d2_s{seed}/validation_rollout_rows.csv"
        table = pd.read_csv(path)
        rows = table.loc[
            table["source"].eq("depablo_mixed_temp") & table["rollout_steps"].eq(100)
        ]
        score = r2(rows["true_p_ratio"].to_numpy(float), rows["pred_p_ratio"].to_numpy(float))
        endpoint_columns_equal = bool(
            np.allclose(rows["true_p_ratio"], rows["endpoint_true_p_ratio"], equal_nan=True)
            and np.allclose(rows["pred_p_ratio"], rows["endpoint_pred_p_ratio"], equal_nan=True)
        )
        per_seed.append({
            "seed": seed,
            "r2": score,
            "valid": int((np.isfinite(rows["true_p_ratio"]) & np.isfinite(rows["pred_p_ratio"])).sum()),
            "total": int(len(rows)),
            "endpoint_columns_equal": endpoint_columns_equal,
        })
        inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    scores = np.asarray([row["r2"] for row in per_seed], dtype=float)
    return {
        "per_seed": per_seed,
        "mean_r2": float(scores.mean()),
        "sample_sd_r2": float(scores.std(ddof=1)),
        "all_endpoint_columns_equal": all(row["endpoint_columns_equal"] for row in per_seed),
        "inputs": inputs,
    }


def mini_endpoint_formula_parity() -> dict:
    usecols = [
        "mode", "source", "frame", "trajectory_id", "true_p_ratio", "pred_p_ratio",
        "true_width", "true_height", "pred_width", "pred_height",
        "true_p_ratio_endpoint", "pred_p_ratio_endpoint", "p_ratio_method",
    ]
    table = pd.read_csv(MINI_ROWS, usecols=usecols)
    table = table.loc[table["source"].isin(["node_mixed", "stiffness_mixed"])].copy()
    keys = ["mode", "source", "trajectory_id"]
    initial = table.loc[table["frame"].eq(0), keys + ["true_width", "true_height"]].rename(
        columns={"true_width": "width0", "true_height": "height0"}
    )
    if initial.duplicated(keys).any():
        raise ValueError("Duplicate frame-zero rows in mini endpoint audit.")
    table = table.merge(initial, on=keys, how="left", validate="many_to_one")
    table["true_recomputed"] = historical_endpoint(
        (table["true_width"] - table["width0"]).to_numpy(float),
        (table["true_height"] - table["height0"]).to_numpy(float),
    )
    table["pred_recomputed"] = historical_endpoint(
        (table["pred_width"] - table["width0"]).to_numpy(float),
        (table["pred_height"] - table["height0"]).to_numpy(float),
    )
    records = []
    score_records = []
    for (mode, source, frame), rows in table.groupby(["mode", "source", "frame"], sort=True):
        true_valid = np.isfinite(rows["true_p_ratio_endpoint"]) & np.isfinite(rows["true_recomputed"])
        pred_valid = np.isfinite(rows["pred_p_ratio_endpoint"]) & np.isfinite(rows["pred_recomputed"])
        records.append({
            "mode": mode,
            "source": source,
            "frame": int(frame),
            "true_valid": int(true_valid.sum()),
            "pred_valid": int(pred_valid.sum()),
            "total": int(len(rows)),
            "true_max_abs_error": float(np.max(np.abs(rows.loc[true_valid, "true_p_ratio_endpoint"] - rows.loc[true_valid, "true_recomputed"]))) if true_valid.any() else None,
            "pred_max_abs_error": float(np.max(np.abs(rows.loc[pred_valid, "pred_p_ratio_endpoint"] - rows.loc[pred_valid, "pred_recomputed"]))) if pred_valid.any() else None,
            "true_nan_pattern_equal": bool(np.array_equal(np.isfinite(rows["true_p_ratio_endpoint"]), np.isfinite(rows["true_recomputed"]))),
            "pred_nan_pattern_equal": bool(np.array_equal(np.isfinite(rows["pred_p_ratio_endpoint"]), np.isfinite(rows["pred_recomputed"]))),
        })
        if frame in (99, 199):
            score_records.append({
                "mode": mode,
                "source": source,
                "frame": int(frame),
                "endpoint_p_ratio_r2": r2(
                    rows["true_p_ratio_endpoint"].to_numpy(float),
                    rows["pred_p_ratio_endpoint"].to_numpy(float),
                ),
                "endpoint_valid": int((
                    np.isfinite(rows["true_p_ratio_endpoint"])
                    & np.isfinite(rows["pred_p_ratio_endpoint"])
                ).sum()),
                "total": int(len(rows)),
            })
    finite_errors = [
        value for row in records for key, value in row.items()
        if key.endswith("max_abs_error") and value is not None and math.isfinite(value)
    ]
    selected = [row for row in records if row["frame"] in (99, 199)]
    selected_errors = [
        value for row in selected for key, value in row.items()
        if key.endswith("max_abs_error") and value is not None and math.isfinite(value)
    ]
    return {
        "rows": int(len(table)),
        "methods": sorted(table["p_ratio_method"].dropna().unique().tolist()),
        "max_abs_error": float(max(finite_errors, default=0.0)),
        "selected_horizon_max_abs_error": float(max(selected_errors, default=0.0)),
        "all_nan_patterns_equal": all(
            row["true_nan_pattern_equal"] and row["pred_nan_pattern_equal"] for row in records
        ),
        "selected_horizons": selected,
        "selected_horizon_scores": score_records,
        "input": {"path": str(MINI_ROWS.relative_to(ROOT)), "sha256": sha256(MINI_ROWS)},
    }


def main() -> None:
    result = {
        "status": "passed",
        "scope": "cached-position metric parity only; no training or model inference",
        "historical_metric": "endpoint fixed-side displacement; min(|-dy/dx|,|-dx/dy|) branch by magnitude",
        "historical_original_limitation": "The exact historical 0.901 checkpoint/code is absent; the preserved five-seed reconstruction is auditable.",
        "old_reconstructed_score": old_score_reconstruction(),
        "mini_endpoint_formula_parity": mini_endpoint_formula_parity(),
        "thermal_ols_is_equivalent": False,
        "thermal_ols_reason": "The OLS estimator fits normalized side strains over a trajectory prefix and applies validity gates; it is a separately labelled metric.",
    }
    old = result["old_reconstructed_score"]
    mini = result["mini_endpoint_formula_parity"]
    # CSV roundoff is amplified at a few very early frames whose deformation
    # is nearly zero. The scientifically reported horizons must agree tightly.
    if (
        not old["all_endpoint_columns_equal"]
        or not mini["all_nan_patterns_equal"]
        or mini["max_abs_error"] > 1e-6
        or mini["selected_horizon_max_abs_error"] > 1e-10
    ):
        result["status"] = "failed"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
