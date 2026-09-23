"""Collect the frozen three-seed global multistep follow-up."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "notebooks/results/graphs_cold_global"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads((HERE / "protocol_multistep.json").read_text())
    split = json.loads((HERE / "split_manifest.json").read_text())
    jobs = json.loads((RESULTS / "multistep_jobs.json").read_text())
    expected = {
        "runner_sha256": sha256(HERE / "run_multistep.py"),
        "protocol_sha256": sha256(HERE / "protocol_multistep.json"),
        "split_manifest_sha256": sha256(HERE / "split_manifest.json"),
    }
    if any(jobs[key] != value for key, value in expected.items()):
        raise RuntimeError("Submitted multistep hashes no longer match the collector inputs.")
    rows = []
    for seed in protocol["seeds"]:
        run_dir = RESULTS / "multistep_global" / str(seed)
        summary = json.loads((run_dir / "summary.json").read_text())
        recipe = json.loads((run_dir / "recipe.json").read_text())
        if any(recipe[key] != value for key, value in expected.items()):
            raise RuntimeError(f"Frozen hash mismatch in {run_dir}")
        if recipe["evaluation_split"] != "final_test" or recipe["evaluation_ids"] != split["final_test"]:
            raise RuntimeError(f"Final-test contract mismatch in {run_dir}")
        for horizon, metrics in summary["test"].items():
            rows.append({"model": "global multistep", "seed": seed, "horizon": int(horizon), **metrics})
    rows.sort(key=lambda row: (row["seed"], row["horizon"]))
    with (RESULTS / "multistep_per_seed_horizon.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for horizon in json.loads((HERE / "protocol.json").read_text())["test"]["horizons"]:
        selected = [row for row in rows if row["horizon"] == horizon]
        result = {"model": "global multistep", "horizon": horizon, "seeds_valid_total": f"{len(selected)}/{len(protocol['seeds'])}"}
        for metric in ("position_mse_physical_minimum_image", "p_ratio_r2_y_over_x", "p_ratio_r2_z_over_x"):
            values = np.asarray([row[metric] for row in selected if row[metric] is not None], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_sd"] = float(values.std(ddof=1))
        for component in ("y", "z"):
            valid = sum(row[f"p_ratio_{component}_valid_total"][0] for row in selected)
            total = sum(row[f"p_ratio_{component}_valid_total"][1] for row in selected)
            result[f"p_ratio_{component}_valid_total"] = f"{valid}/{total}"
        aggregates.append(result)
    with (RESULTS / "multistep_aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    (RESULTS / "multistep_collection.json").write_text(json.dumps({
        "status": "complete",
        "run_count": len(protocol["seeds"]),
        "final_test_networks": len(split["final_test"]),
        "job_ids": jobs["jobs"],
        **expected,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
