#!/usr/bin/env python3
"""Collect the three stride-five propagator seeds and matched stride-one baseline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/mini_temperature200_stride5_pilot"
BASELINE = ROOT / "notebooks/results/mini_temperature200_frames500/all4_twenty_each"
SEEDS = (123, 456, 786)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(truth, prediction) -> float:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    good = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[good], prediction[good]
    denominator = np.square(truth - truth.mean()).sum() if len(truth) else 0.0
    return float(1 - np.square(truth - prediction).sum() / denominator) if len(truth) >= 2 and denominator > 0 else float("nan")


def ols_p(group: pd.DataFrame, prefix: str) -> float:
    x = group[f"{prefix}_boundary_strain_x"].to_numpy(float)
    y = group[f"{prefix}_boundary_strain_y"].to_numpy(float)
    design = np.column_stack((np.ones(len(x)), x))
    return float(-np.linalg.lstsq(design, y, rcond=None)[0][1])


def baseline_summary() -> pd.DataFrame:
    table = pd.read_csv(BASELINE / "test/per_network.csv")
    table = table[table["mode"].eq("autonomous_delta_mlp")]
    rows = []
    for source, source_table in table.groupby("source"):
        truth, prediction = [], []
        for _, group in source_table.groupby("trajectory_id"):
            group = group.sort_values("frame")
            if source.endswith("_mixed"):
                selected = group[group.frame.between(150, 495) & group.frame.mod(5).eq(0)]
                truth.append(ols_p(selected, "true"))
                prediction.append(ols_p(selected, "pred"))
                metric = "strain_OLS_frames150:5:495"
            else:
                selected = group[group.frame.eq(495)].iloc[0]
                truth.append(float(selected.true_p_ratio))
                prediction.append(float(selected.pred_p_ratio))
                metric = "physical_endpoint_frame495"
        rows.append({
            "model": "stride1_200MD",
            "source": source,
            "metric": metric,
            "p_ratio_r2_mean": r2(truth, prediction),
            "p_ratio_r2_sd": float("nan"),
            "seeds": 1,
            "valid_per_seed": len(truth),
            "total_per_seed": len(truth),
        })
    return pd.DataFrame(rows)


def main() -> None:
    seed_tables = []
    hashes = {}
    for seed in SEEDS:
        folder = OUT / f"seed{seed}"
        completion = json.loads((folder / "completed.json").read_text())
        if completion.get("status") != "complete" or completion.get("seed") != seed:
            raise RuntimeError(f"Incomplete seed {seed}")
        table = pd.read_csv(folder / "summary.csv")
        if not table.valid.eq(80).all() or not table.total.eq(80).all():
            raise RuntimeError(f"Invalid evaluation counts for seed {seed}")
        seed_tables.append(table)
        hashes[str(folder / "propagator.pt")] = sha256(folder / "propagator.pt")
    per_seed = pd.concat(seed_tables, ignore_index=True)
    per_seed.to_csv(OUT / "per_seed_summary.csv", index=False)
    aggregate = per_seed.groupby(["source", "metric"], as_index=False).agg(
        p_ratio_r2_mean=("p_ratio_r2", "mean"),
        p_ratio_r2_sd=("p_ratio_r2", "std"),
        p_ratio_mae_mean=("p_ratio_mae", "mean"),
        p_ratio_mae_sd=("p_ratio_mae", "std"),
        seeds=("seed", "nunique"),
        valid_per_seed=("valid", "min"),
        total_per_seed=("total", "max"),
    )
    aggregate.insert(0, "model", "stride5_1000MD")
    comparison = pd.concat((baseline_summary(), aggregate), ignore_index=True)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    receipt = {
        "status": "complete",
        "job_id": os.environ.get("PBS_JOBID"),
        "seeds": list(SEEDS),
        "baseline_ae_sha256": sha256(BASELINE / "ae.pt"),
        "propagator_sha256": hashes,
        "comparison_csv_sha256": sha256(OUT / "comparison.csv"),
        "final_test_used": False,
    }
    (OUT / "completed.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
