"""Aggregate periodic-coordinate p-ratio evaluation across production seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "notebooks/results/graphs_cold_global"


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    rows = []
    for model in protocol["models"]:
        for seed in protocol["seeds"]:
            summary = json.loads((RESULTS / "periodic_evaluation" / model / str(seed) / "summary.json").read_text())
            for horizon, metrics in summary["test"].items():
                rows.append({"model": model, "seed": seed, "horizon": int(horizon), **metrics})
    rows.sort(key=lambda row: (row["model"], row["seed"], row["horizon"]))
    with (RESULTS / "periodic_per_seed_horizon.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for model in protocol["models"]:
        for horizon in protocol["test"]["horizons"]:
            selected = [row for row in rows if row["model"] == model and row["horizon"] == horizon]
            result = {"model": model, "horizon": horizon, "seeds_valid_total": f"{len(selected)}/{len(protocol['seeds'])}"}
            for metric in ("position_mse_physical_minimum_image", "p_ratio_r2_y_over_x", "p_ratio_r2_z_over_x"):
                values = np.asarray([row[metric] for row in selected if row[metric] is not None], dtype=float)
                result[f"{metric}_mean"] = float(values.mean())
                result[f"{metric}_sd"] = float(values.std(ddof=1))
            for component in ("y", "z"):
                valid = sum(row[f"p_ratio_{component}_valid_total"][0] for row in selected)
                total = sum(row[f"p_ratio_{component}_valid_total"][1] for row in selected)
                result[f"p_ratio_{component}_valid_total"] = f"{valid}/{total}"
            aggregates.append(result)
    with (RESULTS / "periodic_aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)


if __name__ == "__main__":
    main()
