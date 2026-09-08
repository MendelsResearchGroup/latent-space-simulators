"""Observed-coordinate motion audit on fixed compact-LJ bridge training IDs.

This is descriptive only: it does not fit a model or read response quantities.
Coordinates use the production per-trajectory reference-box normalization.  The
PCA figures are in-sample, per-network diagnostics and are not AE estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
RESULTS_ROOT = Path(os.environ.get("LSS_RESULTS_ROOT", ROOT / "notebooks/results"))
sys.path.insert(0, str(ROOT / "src"))
from lss.data import load_dataset

MANIFEST = Path(os.environ.get("LSS_SPLIT_MANIFEST", ROOT / "notebooks/results/lj_ae_08_bridge/split_manifest.json"))
OUT = RESULTS_ROOT / "hpc_diagnostics/compact_lj_motion_audit"
SOURCES = ("reid", "depablo_low_temp", "lj_noisy")
RANKS = (2, 4, 6, 8, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(values: list[float]) -> tuple[float | None, int]:
    valid = [float(x) for x in values if np.isfinite(x)]
    return (float(np.mean(valid)) if valid else None), len(valid)


def rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 1e-15 else float("nan")


def trajectory_audit(sim: list, source: str, original_index: int) -> dict:
    positions = np.stack([frame.x[:, :2].detach().cpu().numpy().astype(np.float64) for frame in sim])
    boxes = np.stack([frame.box_tensor[:2].detach().cpu().numpy().astype(np.float64) for frame in sim])
    frames, nodes, _ = positions.shape
    displacement = positions - positions[0]
    increments = np.diff(positions, axis=0)
    # AE inputs are these reference-box-normalized stored coordinates.  For the
    # wrap diagnostic only, compare raw increments with their frame-wise
    # minimum-image equivalents; neither is substituted into the audit metrics.
    minimum_image = increments - np.round(increments / boxes[1:, None, :]) * boxes[1:, None, :]
    wrapped_jump = np.max(np.abs(increments - minimum_image), axis=(1, 2)) > 1e-8
    design = np.concatenate([positions[0], np.ones((nodes, 1))], axis=1)
    affine = []
    for target in positions:
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        affine.append(rms(design @ coef - target))
    centered = positions.reshape(frames, -1) - positions.reshape(frames, -1).mean(axis=0, keepdims=True)
    total_energy = float(np.square(centered).sum())
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    pca = {}
    for rank in RANKS:
        use = min(rank, len(singular))
        reconstruction = (centered @ vh[:use].T) @ vh[:use]
        pca[str(rank)] = float(np.square(centered - reconstruction).sum() / total_energy) if total_energy > 0 else 0.0
    lags = {str(lag): [] for lag in (1, 2, 5, 10)}
    for lag in (1, 2, 5, 10):
        lags[str(lag)] = [cosine(increments[t], increments[t + lag]) for t in range(max(0, len(increments) - lag))]
    zero_error = [rms(frame) for frame in displacement]
    persistence = [rms(value) for value in increments]
    extrapolation = [rms(positions[t] - (2 * positions[t - 1] - positions[t - 2])) for t in range(2, frames)]
    return {
        "source": source, "original_index": int(original_index), "frames": int(frames), "nodes": int(nodes),
        "edges": int(sim[0].edge_index.size(1)), "edge_feature_dim": int(sim[0].edge_attr.size(1)),
        "edge_schema": str(getattr(sim[0], "edge_feature_schema", "")),
        "added_lj_edges": int(getattr(sim[0], "lj_two_hop_edges_added", 0)),
        "nonfinite_coordinate_values": int((~np.isfinite(positions)).sum()),
        "nonfinite_edge_values": int((~np.isfinite(sim[0].edge_attr.detach().cpu().numpy())).sum()),
        "box_min": boxes.min(axis=0).tolist(), "box_max": boxes.max(axis=0).tolist(),
        "coordinate_min": positions.min(axis=(0, 1)).tolist(), "coordinate_max": positions.max(axis=(0, 1)).tolist(),
        "displacement_rms_from_initial_mean": float(np.mean([rms(x) for x in displacement])),
        "increment_rms_mean": float(np.mean(persistence)),
        "translation_mean_node_displacement_rms": float(np.mean([np.linalg.norm(x.mean(axis=0)) for x in displacement])),
        "affine_residual_rms_mean": float(np.mean(affine)),
        "high_frequency_second_difference_over_first": float(rms(np.diff(increments, axis=0)) / rms(increments)) if len(increments) > 1 and rms(increments) > 0 else None,
        "increment_cosine_by_lag": {lag: finite_mean(values)[0] for lag, values in lags.items()},
        "increment_cosine_valid_by_lag": {lag: finite_mean(values)[1] for lag, values in lags.items()},
        "pca_centered_train_reconstruction_error_fraction": pca,
        "zero_displacement_coordinate_rms_mean": float(np.mean(zero_error)),
        "persistence_coordinate_rms_mean": float(np.mean(persistence)),
        "last_increment_extrapolation_coordinate_rms_mean": float(np.mean(extrapolation)),
        "increments_with_minimum_image_difference": int(wrapped_jump.sum()),
        "increment_count": int(len(increments)),
        "stiffness_min": float(sim[0].edge_attr[:, 3].min()), "stiffness_max": float(sim[0].edge_attr[:, 3].max()),
        "lj_indicator_sum": float(sim[0].edge_attr[:, 4].sum()),
    }


def frame_rows(sim: list, source: str, original_index: int) -> list[dict]:
    """Observed-target, in-sample descriptive baselines for each frame."""
    positions = np.stack([f.x[:, :2].detach().cpu().numpy().astype(np.float64) for f in sim])
    frames, nodes, _ = positions.shape
    displacement = positions - positions[0]
    design = np.concatenate([positions[0], np.ones((nodes, 1))], axis=1)
    flattened = positions.reshape(frames, -1)
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    reconstructions = {}
    for rank in (2, 4, 8):
        use = min(rank, vh.shape[0])
        reconstructions[rank] = ((centered @ vh[:use].T) @ vh[:use] + flattened.mean(axis=0)).reshape(positions.shape)
    rows = []
    for t, target in enumerate(positions):
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        row = {"source": source, "original_index": int(original_index), "frame": t, "nodes": nodes,
               "zero_motion_mse": float(np.mean(np.square(displacement[t]))),
               "affine_x0_to_xt_mse": float(np.mean(np.square(design @ coef - target))),
               "pca_in_sample_rank2_mse": float(np.mean(np.square(reconstructions[2][t] - target))),
               "pca_in_sample_rank4_mse": float(np.mean(np.square(reconstructions[4][t] - target))),
               "pca_in_sample_rank8_mse": float(np.mean(np.square(reconstructions[8][t] - target))),
               "persistence_mse": None, "persistence_valid": 0,
               "last_increment_extrapolation_mse": None, "last_increment_extrapolation_valid": 0}
        if t >= 1:
            row.update(persistence_mse=float(np.mean(np.square(positions[t] - positions[t - 1]))), persistence_valid=1)
        if t >= 2:
            row.update(last_increment_extrapolation_mse=float(np.mean(np.square(positions[t] - (2 * positions[t - 1] - positions[t - 2])))), last_increment_extrapolation_valid=1)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-rows-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    for spec in manifest.get("sources", {}).values():
        spec["path"] = str(ROOT / "data" / Path(spec["path"]).name)
    started = time.time()
    rows = []
    per_frame = []
    provenance = {"job_id": os.environ.get("PBS_JOBID", "local"), "status": "running", "manifest": str(MANIFEST),
                  "manifest_sha256": sha256(MANIFEST), "script_sha256": sha256(Path(__file__)),
                  "coordinate_convention": "production position_normalization: each trajectory uses its frame-zero box mapped to [-1,1]^2; stored normalized x is audited without unwrapping",
                  "pca_caveat": "PCA is fitted and scored on all 200 observed training frames of each network separately; it is optimistic and neither a universal AE nor a generalization estimate.",
                  "sources": {}}
    for source in SOURCES:
        spec = manifest["sources"][source]
        ids = list(spec["split_indices"]["train"][:10])
        path = Path(spec["path"])
        actual_hash = sha256(path)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(f"Dataset hash changed for {source}")
        sims = load_dataset(path, coordinate_normalization="position_normalization", append_lj_indicator=True,
                            lj_max_graph_distance=3 if source == "lj_noisy" else None)
        selected = [sims[index] for index in ids]
        if any(len(sim) != 200 for sim in selected):
            raise RuntimeError(f"{source} does not have exactly 200 frames for this audit")
        source_rows = [trajectory_audit(sim, source, idx) for sim, idx in zip(selected, ids)]
        if args.frame_rows_only:
            per_frame.extend(row for sim, idx in zip(selected, ids) for row in frame_rows(sim, source, idx))
        rows.extend(source_rows)
        provenance["sources"][source] = {"path": str(path), "dataset_sha256": actual_hash, "train_ids": ids,
            "count": len(source_rows), "five_channel_schema_verified": all(row["edge_feature_dim"] == 5 for row in source_rows)}
        print(f"audited {source}: {len(source_rows)} trajectories", flush=True)
    if args.frame_rows_only:
        (OUT / "frame_rows_v1.json").write_text(json.dumps(per_frame, indent=2, sort_keys=True) + "\n")
        (OUT / "frame_rows_v1_metadata.json").write_text(json.dumps({"job_id": os.environ.get("PBS_JOBID", "local"), "script_sha256": sha256(Path(__file__)), "data_py_sha256": sha256(ROOT / "src/lss/data.py"), "manifest_sha256": sha256(MANIFEST), "scope": "same first ten train IDs per source as audit 4672395; no validation or reserve trajectory analyzed/evaluated", "baselines": "Affine and PCA are fitted to observed target trajectory frames, so are optimistic descriptive baselines, not predictors or AE ceilings.", "valid_denominators": {"zero_motion": "200 frames/trajectory", "affine": "200 frames/trajectory", "pca": "200 frames/trajectory", "persistence": "199 frames/trajectory", "last_increment_extrapolation": "198 frames/trajectory"}}, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "completed_frame_rows", "rows": len(per_frame)}, indent=2), flush=True)
        return
    (OUT / "trajectory_rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    summary = {}
    numeric = ("displacement_rms_from_initial_mean", "increment_rms_mean", "translation_mean_node_displacement_rms",
               "affine_residual_rms_mean", "high_frequency_second_difference_over_first", "zero_displacement_coordinate_rms_mean",
               "persistence_coordinate_rms_mean", "last_increment_extrapolation_coordinate_rms_mean", "increments_with_minimum_image_difference")
    for source in SOURCES:
        group = [row for row in rows if row["source"] == source]
        summary[source] = {"trajectories_valid": len(group), "trajectories_total": 10,
            **{key: {"mean": finite_mean([row[key] for row in group])[0], "valid": finite_mean([row[key] for row in group])[1], "total": len(group)} for key in numeric},
            "increment_cosine_by_lag": {lag: {"mean": finite_mean([row["increment_cosine_by_lag"][lag] for row in group])[0], "valid": finite_mean([row["increment_cosine_by_lag"][lag] for row in group])[1], "total": len(group)} for lag in ("1", "2", "5", "10")},
            "pca_error_fraction_by_rank": {rank: {"mean": float(np.mean([row["pca_centered_train_reconstruction_error_fraction"][rank] for row in group])), "valid": len(group), "total": len(group)} for rank in map(str, RANKS)}}
    provenance.update({"status": "completed", "seconds": time.time() - started, "trajectory_rows": len(rows), "summary": summary,
        "limitations": ["Coordinate jumps are reported only as raw-versus-minimum-image increment differences; they are not called noise.", "Temporal correlations and in-sample PCA reconstruction cannot establish irreducible noise, causation, or an AE's achievable generalization.", "Only the fixed first ten training IDs per source were used; no validation, mixed-temperature, or final-test trajectories were accessed."]})
    (OUT / "audit.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "seconds": provenance["seconds"], "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
