"""Collect the frozen cold-graph local-versus-global production matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "notebooks/results/graphs_cold_global"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    split = json.loads((HERE / "split_manifest.json").read_text())
    jobs = json.loads((RESULTS / "production_jobs.json").read_text())
    expected_hashes = {
        "protocol_sha256": jobs["protocol_sha256"],
        "split_manifest_sha256": jobs["split_manifest_sha256"],
        "runner_sha256": jobs["runner_sha256"],
    }
    current_hashes = {
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "split_manifest_sha256": sha256(HERE / "split_manifest.json"),
        "runner_sha256": sha256(HERE / "run.py"),
    }
    if current_hashes != expected_hashes:
        raise RuntimeError("Frozen production hashes no longer match the submitted matrix.")

    run_rows = []
    for model in protocol["models"]:
        for seed in protocol["seeds"]:
            run_dir = RESULTS / "production" / model / str(seed)
            summary_path = run_dir / "summary.json"
            recipe_path = run_dir / "recipe.json"
            if not summary_path.is_file() or not recipe_path.is_file():
                raise FileNotFoundError(f"Incomplete production run: {run_dir}")
            summary = json.loads(summary_path.read_text())
            recipe = json.loads(recipe_path.read_text())
            if any(recipe[key] != value for key, value in expected_hashes.items()):
                raise RuntimeError(f"Frozen hash mismatch in {recipe_path}")
            if recipe["evaluation_split"] != "final_test" or recipe["evaluation_ids"] != split["final_test"]:
                raise RuntimeError(f"Final-test contract mismatch in {recipe_path}")
            for horizon, metrics in summary["test"].items():
                run_rows.append({
                    "model": model,
                    "seed": seed,
                    "horizon": int(horizon),
                    "epochs_completed": summary["epochs_completed"],
                    "best_validation_one_step_combined_state_mse": summary["best_validation_one_step_combined_state_mse"],
                    **metrics,
                })

    run_rows.sort(key=lambda row: (row["model"], row["seed"], row["horizon"]))
    with (RESULTS / "per_seed_horizon.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    aggregate_rows = []
    for model in protocol["models"]:
        for horizon in protocol["test"]["horizons"]:
            rows = [row for row in run_rows if row["model"] == model and row["horizon"] == horizon]
            aggregate = {"model": model, "horizon": horizon, "seeds_valid_total": f"{len(rows)}/{len(protocol['seeds'])}"}
            for metric in (
                "position_mse_physical_minimum_image",
                "p_ratio_r2_y_over_x",
                "p_ratio_r2_z_over_x",
            ):
                values = np.asarray([row[metric] for row in rows if row[metric] is not None], dtype=float)
                aggregate[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
                aggregate[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            for component in ("y", "z"):
                valid = sum(row[f"p_ratio_{component}_valid_total"][0] for row in rows)
                total = sum(row[f"p_ratio_{component}_valid_total"][1] for row in rows)
                aggregate[f"p_ratio_{component}_valid_total"] = f"{valid}/{total}"
            aggregate_rows.append(aggregate)

    with (RESULTS / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (RESULTS / "collection.json").write_text(json.dumps({
        "status": "complete",
        "run_count": len(protocol["models"]) * len(protocol["seeds"]),
        "final_test_networks": len(split["final_test"]),
        "job_ids": jobs["jobs"],
        **expected_hashes,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
