"""Validate and collect the completed latent-count matrix using direct z0/z1 metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "notebooks/results/latent_count_morph"
MATRIX_PATH = ROOT / "hpc/experiments/latent_count_morph/matrix.json"
RUNNER_PATH = ROOT / "hpc/experiments/latent_count_morph/run.py"
OUT = BASE / "collected"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    matrix = json.loads(MATRIX_PATH.read_text())
    runner_hash = sha256(RUNNER_PATH)
    expected_data_hash = "c92162d0c2fd475e41f13f41596685274b346e7d81d23e9115aba317948c824a"
    metric_rows = []
    job_rows = []
    all_initial = []
    all_trajectories = []
    for index, cell in enumerate(matrix):
        count, seed = int(cell["count"]), int(cell["seed"])
        cell_dir = BASE / f"c{count}_s{seed}"
        required = [cell_dir / "completed.json", cell_dir / "initial_latents.csv", cell_dir / "trajectory_latents.csv"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing completed matrix artifacts:\n" + "\n".join(missing))
        completed = json.loads(required[0].read_text())
        checks = {
            "status": completed["status"] == "completed",
            "count": int(completed["count"]) == count,
            "seed": int(completed["seed"]) == seed,
            "runner": completed["script_sha256"] == runner_hash,
            "data": completed["data_manifest_sha256"] == expected_data_hash,
        }
        if not all(checks.values()):
            raise ValueError(f"Invalid matrix cell {index}: {checks}")
        initial = pd.read_csv(required[1])
        trajectories = pd.read_csv(required[2])
        if len(initial) != 160 or not initial.groupby("source").size().eq(80).all():
            raise ValueError(f"Cell {index}: invalid initial latent counts")
        if len(trajectories) != 3200 or not trajectories.groupby(["source", "trajectory_id"]).size().eq(200).all():
            raise ValueError(f"Cell {index}: invalid trajectory latent counts")
        initial.insert(0, "seed", seed); initial.insert(0, "count", count)
        trajectories.insert(0, "seed", seed); trajectories.insert(0, "count", count)
        all_initial.append(initial); all_trajectories.append(trajectories)
        for source, group in initial.groupby("source"):
            z = group[["z0", "z1"]].to_numpy(float)
            corr = float(group.z0.corr(group.z1))
            singular = np.linalg.svd(z - z.mean(0), compute_uv=False)
            eigenvalues = singular**2
            path_group = trajectories[trajectories.source.eq(source)]
            early_angles = {1: [], 10: [], 20: []}
            for _, path in path_group.groupby("trajectory_id"):
                path = path.sort_values("frame")
                values = path[["z0", "z1"]].to_numpy(float)
                displacement = values - values[0]
                final = displacement[-1]
                final_norm = np.linalg.norm(final)
                for frame in early_angles:
                    current = displacement[frame]
                    denominator = np.linalg.norm(current) * final_norm
                    cosine = np.clip(current @ final / denominator, -1, 1) if denominator > 1e-12 else np.nan
                    early_angles[frame].append(np.degrees(np.arccos(cosine)))
            metric_rows.append({
                "count": count,
                "seed": seed,
                "source": source,
                "corr_initial_z0_z1": corr,
                "shared_initial_variation_r2": corr**2,
                "initial_effective_dimension": float(eigenvalues.sum()**2 / np.square(eigenvalues).sum()),
                "early_angle_frame1_median": float(np.nanmedian(early_angles[1])),
                "early_angle_frame10_median": float(np.nanmedian(early_angles[10])),
                "early_angle_frame20_median": float(np.nanmedian(early_angles[20])),
                "initial_valid": len(group),
                "initial_total": 80,
                "trajectory_valid": path_group.trajectory_id.nunique(),
                "trajectory_total": 8,
            })
        job_rows.append({
            "matrix_index": index,
            "count": count,
            "seed": seed,
            "job_id": completed["job_id"],
            "reused_existing_checkpoint": completed["reused_existing_checkpoint"],
            "checkpoint_sha256": completed["checkpoint_sha256"],
            "seconds": completed["seconds"],
            "max_rss_kib": completed["max_rss_kib"],
            "status": completed["status"],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["source", "count", "seed"])
    metrics.to_csv(OUT / "raw_z_metrics.csv", index=False)
    pd.DataFrame(job_rows).sort_values("matrix_index").to_csv(OUT / "cells.csv", index=False)
    pd.concat(all_initial, ignore_index=True).to_csv(OUT / "initial_latents.csv", index=False)
    pd.concat(all_trajectories, ignore_index=True).to_csv(OUT / "trajectory_latents.csv", index=False)
    aggregate = metrics.groupby(["source", "count"]).agg(
        seeds=("seed", "nunique"),
        shared_initial_variation_mean=("shared_initial_variation_r2", "mean"),
        shared_initial_variation_sd=("shared_initial_variation_r2", "std"),
        effective_dimension_mean=("initial_effective_dimension", "mean"),
        effective_dimension_sd=("initial_effective_dimension", "std"),
        early_angle_frame10_mean=("early_angle_frame10_median", "mean"),
        early_angle_frame10_sd=("early_angle_frame10_median", "std"),
        initial_valid=("initial_valid", "sum"),
        initial_total=("initial_total", "sum"),
        trajectory_valid=("trajectory_valid", "sum"),
        trajectory_total=("trajectory_total", "sum"),
    ).reset_index()
    aggregate.to_csv(OUT / "aggregate.csv", index=False)
    metadata = {
        "status": "complete",
        "matrix_cells_valid": len(matrix),
        "matrix_cells_total": 24,
        "counts": sorted(metrics["count"].unique().tolist()),
        "seeds": sorted(metrics["seed"].unique().tolist()),
        "sources": sorted(metrics["source"].unique().tolist()),
        "runner_sha256": runner_hash,
        "matrix_sha256": sha256(MATRIX_PATH),
        "data_manifest_sha256": expected_data_hash,
        "collector_sha256": sha256(Path(__file__)),
        "coordinate_rule": "raw z0/z1 retained per model/seed; no PCA coordinate replacement or response-based alignment",
        "response_use": "none in collected morphology metrics",
        "early_motion": "fixed response-blind trajectory IDs 40,50,...110; angle of z(frame)-z(0) to z(199)-z(0)",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(aggregate.round(3).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
