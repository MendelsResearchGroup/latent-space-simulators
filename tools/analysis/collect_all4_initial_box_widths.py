"""Cache initial boundary widths for the frozen all4 held-out cohort."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from graph_utils import directional_side_indices_from_box


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "notebooks/results/all4_coverage_1000"
MANIFEST = BASE / "data_manifest.json"
OUTPUT = BASE / "all4_twenty_each_retry_v5_validation30/test/initial_box_widths.csv"
SOURCES = ("node_low", "stiffness_low")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    rows: list[dict[str, float | int | str]] = []
    for source in SOURCES:
        entry = manifest["datasets"][source]
        dataset_path = Path(entry["test_path"])
        if sha256(dataset_path) != entry["test_sha256"]:
            raise ValueError(f"Dataset hash mismatch: {dataset_path}")
        simulations = torch.load(dataset_path, map_location="cpu", weights_only=False)
        for trajectory_id in range(40, 120):
            reference = simulations[trajectory_id][0]
            sides = directional_side_indices_from_box(reference, quantile=0.10)
            xy = reference.x[:, :2].detach().cpu().numpy()
            initial_lx = xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean()
            initial_ly = xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean()
            rows.append(
                {
                    "source": source,
                    "trajectory_id": trajectory_id,
                    "initial_Lx": float(initial_lx),
                    "initial_Ly": float(initial_ly),
                }
            )
        del simulations
        gc.collect()
    table = pd.DataFrame(rows)
    if len(table) != 160 or table.duplicated(["source", "trajectory_id"]).any():
        raise ValueError("Expected 160 unique held-out source/trajectory rows.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(table)} rows")


if __name__ == "__main__":
    main()
