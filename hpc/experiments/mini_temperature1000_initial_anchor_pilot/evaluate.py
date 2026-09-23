"""Post-fit 30-network mixed-T comparison for the initial-anchor pilot."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from anchor_context import install

install()

from lss.dynamics.capacity import load_experiment_bundle

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE_STUDY = ROOT / "notebooks/results/mini_temperature1000_mixed_only"
BASE = BASE_STUDY / "joint_mixed_only"
OUT = ROOT / "notebooks/results/mini_temperature1000_initial_anchor_pilot"
ORIGINAL_CODE = ROOT / "hpc/experiments/mini_temperature1000_mixed_only"
sys.path.insert(0, str(ORIGINAL_CODE))
import linear_run as original_linear


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    denominator = np.square(truth - truth.mean()).sum()
    return float(1 - np.square(truth - prediction).sum() / denominator)


def summarize(table: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for source, group in table.query("mode=='autonomous_delta_mlp'").groupby("source"):
        linear = group.dropna(subset=["linear_true_p", "linear_pred_p"])
        endpoint = group.dropna(subset=["endpoint_true_p", "endpoint_pred_p"])
        rows.append({
            "model": label,
            "source": source,
            "linear_fit_p_ratio_r2": r2(linear.linear_true_p, linear.linear_pred_p),
            "linear_fit_valid": len(linear),
            "linear_fit_total": len(group),
            "frame199_endpoint_p_ratio_r2": r2(endpoint.endpoint_true_p, endpoint.endpoint_pred_p),
            "endpoint_valid": len(endpoint),
            "endpoint_total": len(group),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    destination = OUT / ("anchored_s123_smoke" if args.smoke else "anchored_s123")
    if json.loads((destination / "fit_completed.json").read_text())["status"] != "completed":
        raise RuntimeError("Fit is not complete.")
    recipe = json.loads((destination / "recipe.json").read_text())
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    result = load_experiment_bundle(destination / "propagator.pt", cfg=recipe["config"], device="cpu")
    manifest = json.loads((BASE_STUDY / "data_manifest.json").read_text())
    rows = []
    count = 1 if args.smoke else 30
    for source in ("node_mixed", "stiffness_mixed"):
        sims = original_linear.sims(result, manifest["datasets"][source], source)[:count]
        rows.extend(original_linear.eval_source(result, sims, source, "initial_anchor"))
    table = pd.DataFrame(rows)
    table.to_csv(destination / "per_network.csv", index=False)
    if args.smoke:
        (destination / "completed.json").write_text(json.dumps({
            "status": "completed", "smoke": True, "networks_per_source": 1
        }, indent=2) + "\n")
        return
    baseline = pd.read_csv(BASE_STUDY / "linear_fit/joint_mixed_only/per_network.csv")
    baseline = baseline[baseline.trajectory_id.between(40, 69)].copy()
    comparison = pd.DataFrame(summarize(baseline, "baseline") + summarize(table, "initial anchor"))
    comparison.to_csv(destination / "comparison.csv", index=False)
    (destination / "completed.json").write_text(json.dumps({
        "status": "completed",
        "job_id": os.environ.get("PBS_JOBID", "local"),
        "networks_per_source": 30,
        "heldout_ids": list(range(40, 70)),
        "baseline_ae_sha256": sha(BASE / "ae.pt"),
        "propagator_sha256": sha(destination / "propagator.pt"),
        "primary_metric": "post-fit p-ratio R2; normalized boundary-strain OLS frames30..199",
        "secondary_metric": "physical frame199 endpoint p-ratio R2",
        "test_used_for_selection": False,
    }, indent=2) + "\n")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

