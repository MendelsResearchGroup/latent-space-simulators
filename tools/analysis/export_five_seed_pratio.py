#!/usr/bin/env python3
"""Estimate boundary-strain p-ratio from five matched MetaForge replicas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
METAFORGE = ROOT.parent / "MetaForge"
for path in (ROOT / "src", METAFORGE / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph_utils import directional_side_indices_from_box  # noqa: E402
from lss.dynamics.simulation import replicate_fixed_effects_p_ratio  # noqa: E402


SEEDS = (104729, 224737, 350377, 479909, 611953)
ASSIGNMENT_SEED = 904297
BRANCHES = {
    "node": "mini_node_optimized_200_500",
    "stiffness": "mini_stiff_optimized_200_1500",
}
CADENCES = {
    "1000md": {
        "root": METAFORGE / "results/mini_temperature_1000_replicates",
        "frames": 200,
        "sample_every": 1000,
        "windows": (("standard", 30, 199),),
    },
    "200md": {
        "root": METAFORGE / "results/mini_temperature_200_1500frames_replicates",
        "frames": 1500,
        "sample_every": 200,
        "windows": (
            ("short_100k", 150, 499),
            ("matched_199k", 150, 995),
            ("full_300k", 150, 1499),
        ),
    },
}
PROFILES = ("low", "mixed")
T95_DF4 = 2.7764451051977987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifests(cadence: str, branch: str, profile: str) -> list[dict]:
    spec = CADENCES[cadence]
    manifests = []
    for seed in SEEDS:
        directory = spec["root"] / f"seed_{seed}" / profile / BRANCHES[branch]
        manifest_path = directory / "manifest.json"
        recipe_path = directory / "recipe.json"
        manifest = json.loads(manifest_path.read_text())
        recipe = json.loads(recipe_path.read_text())
        if manifest.get("status") != "complete" or len(manifest.get("records", [])) != 200:
            raise RuntimeError(f"Incomplete replicate: {directory}")
        if manifest.get("frames_per_trajectory") != spec["frames"]:
            raise RuntimeError(f"Unexpected frame count: {directory}")
        if manifest.get("md_steps_between_frames") != spec["sample_every"]:
            raise RuntimeError(f"Unexpected sampling interval: {directory}")
        if recipe.get("thermal_seed_base") != seed:
            raise RuntimeError(f"Thermostat seed mismatch: {directory}")
        if recipe.get("assignment_seed") != ASSIGNMENT_SEED:
            raise RuntimeError(f"Temperature assignment mismatch: {directory}")
        manifest["_path"] = str(manifest_path)
        manifest["_sha256"] = sha256(manifest_path)
        manifests.append(manifest)
    return manifests


def assert_identity(records: list[dict], branch: str, profile: str, index: int) -> None:
    expected = records[0]
    fields = (
        "source_index",
        "source_chunk",
        "temperature_K",
        "recorded_ground_truth_p_ratio",
        "source_rollout_p_ratio",
    )
    identity = tuple(expected[field] for field in fields)
    for seed, record in zip(SEEDS, records):
        if tuple(record[field] for field in fields) != identity:
            raise RuntimeError(f"Identity mismatch: {branch}/{profile}/{index}/seed_{seed}")
        if record["thermal_seed"] != seed + 1 + index:
            raise RuntimeError(f"Thermal seed mismatch: {branch}/{profile}/{index}/seed_{seed}")


def dimensions(trajectory: list, sides: dict[str, np.ndarray]) -> np.ndarray:
    values = np.empty((len(trajectory), 2), dtype=float)
    for frame, graph in enumerate(trajectory):
        xy = graph.x[:, :2].detach().cpu().numpy()
        values[frame, 0] = float(
            xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean()
        )
        values[frame, 1] = float(
            xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean()
        )
    return values


def strains_for_records(records: list[dict], reference_record: dict) -> tuple[np.ndarray, np.ndarray, bool]:
    reference = torch.load(reference_record["trajectory"], map_location="cpu", weights_only=False)
    sides = directional_side_indices_from_box(reference[0], quantile=0.10)
    reference_membership = {key: tuple(value.tolist()) for key, value in sides.items()}
    del reference

    all_x = []
    all_y = []
    memberships_match = True
    for record in records:
        trajectory = torch.load(record["trajectory"], map_location="cpu", weights_only=False)
        current_sides = directional_side_indices_from_box(trajectory[0], quantile=0.10)
        memberships_match &= all(
            tuple(current_sides[key].tolist()) == reference_membership[key]
            for key in reference_membership
        )
        wh = dimensions(trajectory, sides)
        if not np.isfinite(wh).all() or np.any(np.abs(wh[0]) <= 1e-12):
            raise RuntimeError(f"Invalid boundary dimensions: {record['trajectory']}")
        strain = (wh - wh[0]) / wh[0]
        all_x.append(strain[:, 0])
        all_y.append(strain[:, 1])
        del trajectory
    return np.stack(all_x), np.stack(all_y), memberships_match


def endpoint_summary(x: np.ndarray, y: np.ndarray, frame: int) -> dict[str, float]:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = -y[:, frame] / x[:, frame]
        ensemble = -float(y[:, frame].mean() / x[:, frame].mean())
    finite = values[np.isfinite(values)]
    return {
        "endpoint_ensemble_p_ratio": ensemble,
        "endpoint_replicate_mean_p_ratio": float(finite.mean()) if len(finite) else float("nan"),
        "endpoint_replicate_sd_p_ratio": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
    }


def fitted_row(x: np.ndarray, y: np.ndarray, start: int, stop: int) -> dict[str, object]:
    fit = replicate_fixed_effects_p_ratio(x[:, start : stop + 1], y[:, start : stop + 1])
    per_seed = np.asarray(fit.pop("per_replicate_p_ratio"), dtype=float)
    row = dict(fit)
    for seed, value in zip(SEEDS, per_seed):
        row[f"p_ratio_seed_{seed}"] = float(value)
    sem = float(row["replicate_sem_p_ratio"])
    row["replicate_ci95_low"] = float(row["replicate_mean_p_ratio"] - T95_DF4 * sem)
    row["replicate_ci95_high"] = float(row["replicate_mean_p_ratio"] + T95_DF4 * sem)
    return row


def prefix_horizons(cadence: str) -> list[int]:
    spec = CADENCES[cadence]
    first = 30_000 // spec["sample_every"]
    step = max(1, 10_000 // spec["sample_every"])
    horizons = set(range(first + step, spec["frames"], step))
    horizons.add(spec["frames"] - 1)
    if cadence == "200md":
        horizons.update((499, 995))
    return sorted(horizons)


def export(cadence: str, branch: str, profile: str, output: Path) -> None:
    manifests = load_manifests(cadence, branch, profile)
    low_manifests = manifests if profile == "low" else load_manifests(cadence, branch, "low")
    spec = CADENCES[cadence]
    network_rows = []
    prefix_rows = []
    membership_mismatches = 0

    for index in range(200):
        records = [manifest["records"][index] for manifest in manifests]
        assert_identity(records, branch, profile, index)
        reference_record = low_manifests[0]["records"][index]
        x, y, membership_match = strains_for_records(records, reference_record)
        membership_mismatches += int(not membership_match)
        identity = records[0]
        base = {
            "cadence": cadence,
            "md_steps_between_frames": spec["sample_every"],
            "branch": branch,
            "profile": profile,
            "source_index": index,
            "source_chunk": identity["source_chunk"],
            "temperature_K": float(identity["temperature_K"]),
            "recorded_ground_truth_p_ratio": identity["recorded_ground_truth_p_ratio"],
            "source_rollout_p_ratio": identity["source_rollout_p_ratio"],
            "side_memberships_match_reference": bool(membership_match),
        }
        for window, start, stop in spec["windows"]:
            network_rows.append(
                {
                    **base,
                    "window": window,
                    "start_frame": start,
                    "stop_frame": stop,
                    "start_md_step": start * spec["sample_every"],
                    "stop_md_step": stop * spec["sample_every"],
                    **fitted_row(x, y, start, stop),
                    **endpoint_summary(x, y, stop),
                }
            )
        start = 30_000 // spec["sample_every"]
        for stop in prefix_horizons(cadence):
            prefix_rows.append(
                {
                    **base,
                    "start_frame": start,
                    "stop_frame": stop,
                    "start_md_step": start * spec["sample_every"],
                    "stop_md_step": stop * spec["sample_every"],
                    **fitted_row(x, y, start, stop),
                    **endpoint_summary(x, y, stop),
                }
            )
        if (index + 1) % 10 == 0:
            print(f"{cadence}/{branch}/{profile}: {index + 1}/200", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(network_rows).to_csv(output / "per_network_window.csv", index=False)
    pd.DataFrame(prefix_rows).to_csv(output / "per_network_prefix.csv", index=False)
    metadata = {
        "status": "complete",
        "job_id": os.environ.get("PBS_JOBID"),
        "cadence": cadence,
        "branch": branch,
        "profile": profile,
        "networks": 200,
        "replicates": len(SEEDS),
        "thermostat_seed_bases": list(SEEDS),
        "temperature_assignment_seed": ASSIGNMENT_SEED,
        "frames_per_trajectory": spec["frames"],
        "md_steps_between_frames": spec["sample_every"],
        "estimator": "fixed-effects OLS of normalized transverse boundary strain on normalized axial boundary strain; one intercept per thermal seed and one shared slope; p=-slope",
        "boundary": "fixed frame-0 q=0.10 side nodes from the first low-T replica of the same network",
        "replica_uncertainty": "single-replica OLS mean, sample SD, SEM, and two-sided 95% t interval with df=4",
        "membership_mismatch_networks": membership_mismatches,
        "manifest_sha256": {manifest["_path"]: manifest["_sha256"] for manifest in manifests},
        "script_sha256": sha256(Path(__file__)),
        "final_test_used": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadence", choices=tuple(CADENCES), required=True)
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    export(args.cadence, args.branch, args.profile, args.output)


if __name__ == "__main__":
    main()
