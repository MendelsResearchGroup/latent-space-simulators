#!/usr/bin/env python3
"""Post-fit mixed-T horizon audit for the dense 200-MD-step rollout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "notebooks/results/mini_temperature200_frames500/all4_twenty_each"
OUT = ROOT / "notebooks/results/mini_temperature200_frames500/rollout_metric_audit"
SOURCES = ("node_mixed", "stiffness_mixed")
CUMULATIVE_START = 150
MIN_PAIRS = 8
FIXED_WINDOW = 150
FINAL_HORIZON = 499


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(truth, prediction) -> float:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    denominator = np.square(truth - truth.mean()).sum()
    return float(1 - np.square(truth - prediction).sum()/denominator) if denominator > 0 else float("nan")


def fit_p(x, y) -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() != len(x) or len(x) < MIN_PAIRS or np.square(x-x.mean()).sum() <= 0:
        return float("nan"), float("nan")
    design = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(-beta[1]), r2(y, design@beta)


def main() -> None:
    assert not OUT.exists(), OUT
    OUT.mkdir()
    path = MODEL / "test/per_network.csv"
    checkpoint_before = {name: sha(MODEL/name) for name in ("ae.pt", "propagator.pt")}
    data = pd.read_csv(path).query("source in @SOURCES and mode == 'autonomous_delta_mlp'")
    rows = []
    for (source, trajectory_id), group in data.groupby(["source", "trajectory_id"]):
        group = group.sort_values("frame")
        assert group.frame.tolist() == list(range(500))
        tx = group.true_boundary_strain_x.to_numpy()
        ty = group.true_boundary_strain_y.to_numpy()
        px = group.pred_boundary_strain_x.to_numpy()
        py = group.pred_boundary_strain_y.to_numpy()
        final_true_p, final_true_fit_r2 = fit_p(tx[CUMULATIVE_START:500], ty[CUMULATIVE_START:500])
        for horizon in range(CUMULATIVE_START + MIN_PAIRS - 1, FINAL_HORIZON + 1):
            cumulative_true_p, cumulative_true_fit_r2 = fit_p(tx[CUMULATIVE_START:horizon+1], ty[CUMULATIVE_START:horizon+1])
            cumulative_pred_p, cumulative_pred_fit_r2 = fit_p(px[CUMULATIVE_START:horizon+1], py[CUMULATIVE_START:horizon+1])
            row = {"source": source, "trajectory_id": int(trajectory_id), "horizon": horizon,
                   "md_step": horizon*200, "final_true_p": final_true_p,
                   "final_true_fit_r2": final_true_fit_r2,
                   "cumulative_true_p": cumulative_true_p, "cumulative_pred_p": cumulative_pred_p,
                   "cumulative_true_fit_r2": cumulative_true_fit_r2,
                   "cumulative_pred_fit_r2": cumulative_pred_fit_r2}
            if horizon >= CUMULATIVE_START + FIXED_WINDOW - 1:
                start = horizon - FIXED_WINDOW + 1
                row["fixed_start"] = start
                row["fixed_true_p"], row["fixed_true_fit_r2"] = fit_p(tx[start:horizon+1], ty[start:horizon+1])
                row["fixed_pred_p"], row["fixed_pred_fit_r2"] = fit_p(px[start:horizon+1], py[start:horizon+1])
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "per_network.csv", index=False)
    summary = []
    for (source, horizon), group in table.groupby(["source", "horizon"]):
        row = {"source": source, "horizon": horizon, "md_step": horizon*200,
            "cumulative_same_horizon_r2": r2(group.cumulative_true_p, group.cumulative_pred_p),
            "cumulative_truth_vs_final_r2": r2(group.final_true_p, group.cumulative_true_p),
            "cumulative_pred_vs_final_r2": r2(group.final_true_p, group.cumulative_pred_p),
            "cumulative_true_fit_r2_median": float(group.cumulative_true_fit_r2.median()),
            "cumulative_pred_fit_r2_median": float(group.cumulative_pred_fit_r2.median()),
            "valid": len(group), "total": len(group)}
        fixed = group.dropna(subset=["fixed_true_p", "fixed_pred_p"])
        if len(fixed):
            row.update({"fixed_same_horizon_r2": r2(fixed.fixed_true_p, fixed.fixed_pred_p),
                "fixed_truth_vs_final_r2": r2(fixed.final_true_p, fixed.fixed_true_p),
                "fixed_pred_vs_final_r2": r2(fixed.final_true_p, fixed.fixed_pred_p),
                "fixed_true_fit_r2_median": float(fixed.fixed_true_fit_r2.median()),
                "fixed_pred_fit_r2_median": float(fixed.fixed_pred_fit_r2.median())})
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    checkpoint_after = {name: sha(MODEL/name) for name in ("ae.pt", "propagator.pt")}
    assert checkpoint_before == checkpoint_after
    receipt = {"status": "completed", "postfit_only": True, "sources": list(SOURCES),
        "heldout_ids": [40,119], "cumulative_fit_frames": [150,499],
        "fixed_window_frames": FIXED_WINDOW, "fixed_window_md_span": (FIXED_WINDOW-1)*200,
        "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after,
        "script_sha256": sha(Path(__file__))}
    (OUT/"completed.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
