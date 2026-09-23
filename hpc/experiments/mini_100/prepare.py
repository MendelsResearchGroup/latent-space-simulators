#!/usr/bin/env python3
"""Materialize exact first-100-frame cohorts from canonical full mini data."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
DATA = ROOT / "data"
RESULTS = ROOT / "notebooks/results/mini_100"
COUNT, FRAMES = 120, 100


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_prefix(dataset, expected_paths: list[str], category: str):
    if len(dataset) < COUNT or len(expected_paths) < COUNT:
        raise ValueError(f"{category}: fewer than {COUNT} trajectories")
    selected = dataset[:COUNT]
    actual_paths = []
    for trajectory_index, trajectory in enumerate(selected):
        if len(trajectory) < FRAMES:
            raise ValueError(f"{category} trajectory {trajectory_index}: only {len(trajectory)} frames")
        first = trajectory[0]
        expected_chunk = Path(expected_paths[trajectory_index]).name
        if str(first.source_chunk) != expected_chunk:
            raise ValueError(f"{category} trajectory {trajectory_index}: source chunk order mismatch")
        topology, stiffness = first.edge_index, first.edge_attr[:, 3]
        for frame_index, frame in enumerate(trajectory[:FRAMES]):
            if int(frame.source_frame_index) != frame_index:
                raise ValueError(f"{category} trajectory {trajectory_index}: frame index mismatch")
            if str(frame.source_chunk) != expected_chunk:
                raise ValueError(f"{category} trajectory {trajectory_index}: source chunk changes")
            if not torch.equal(frame.edge_index, topology) or not torch.equal(frame.edge_attr[:, 3], stiffness):
                raise ValueError(f"{category} trajectory {trajectory_index}: static graph changes")
        actual_paths.append(expected_paths[trajectory_index])
    return [trajectory[:FRAMES] for trajectory in selected], actual_paths


def source_spec(base_source: dict, *, name: str, label: str, path: Path) -> dict:
    source = copy.deepcopy(base_source)
    source.update(name=name, source_name=name, dataset_name=name, label=label, path=str(path.resolve()))
    mixture = source["dataset_mixture"][0]
    mixture.update(name=name, label=label, path=str(path.resolve()), train_count=20, val_count=20,
                   split_indices={"train": list(range(20)), "val": list(range(20, 40)), "test": []})
    return source


def write_prepared(name: str, source: dict, output: Path, hashes: dict, selected_paths: list[str]) -> None:
    payload = {
        "source": source, "prefix_sha256": sha(output), "evaluation_path": str(output.resolve()),
        "evaluation_sha256": sha(output), "test_indices": list(range(40, 120)),
        "split": {"train": 20, "validation": 20, "test": 80}, "frames": list(range(FRAMES)),
        "selected_source_paths": selected_paths, "provenance_hashes": hashes,
    }
    (RESULTS / f"{name}_prepared.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    old_prepared = ROOT / "notebooks/results/mini_stiffness/prepared.json"
    prefix_manifest = DATA / "mini_data_converted.manifest.json"
    full_manifest_path = DATA / "mini_data_full200.manifest.json"
    full_manifest = json.loads(full_manifest_path.read_text())
    base_source = json.loads(old_prepared.read_text())["source"]
    stiff_paths = full_manifest["selected_paths"]["stiff_optimized"]
    node_paths = full_manifest["selected_paths"]["node_optimized"]
    prefix = json.loads(prefix_manifest.read_text())
    expected_old = [prefix["records"]["stiff_optimized"][index]["file_path"].removeprefix("./")
                    for index in prefix["stiffness_cohort_record_indices"]]
    if stiff_paths[:COUNT] != expected_old:
        raise ValueError("full stiffness order does not preserve existing 120 cohort")
    code_hash = os.environ["LSS_PREP_CODE_SHA256"]
    pbs_hash = os.environ["LSS_PREP_PBS_SHA256"]
    common = {"old_prepared": sha(old_prepared), "prefix_manifest": sha(prefix_manifest),
              "full_manifest": sha(full_manifest_path), "preparation_code": code_hash, "prepare_pbs": pbs_hash}
    for category, dataset_path, selected_paths, name, label in (
        ("stiff_optimized", DATA / "mini_stiff_optimized_200_1500.pt", stiff_paths, "mini_stiffness", "Stiffness-optimized mini networks"),
        ("node_optimized", DATA / "mini_node_optimized_200_500.pt", node_paths, "mini_node", "Node-optimized mini networks"),
    ):
        dataset_hash = sha(dataset_path)
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
        subset, selected = validate_and_prefix(dataset, selected_paths, category)
        output = DATA / f"{name}_cohort120_first100.pt"
        temporary = output.with_suffix(output.suffix + ".partial")
        torch.save(subset, temporary)
        temporary.replace(output)
        hashes = {**common, "full_dataset": dataset_hash, "output": sha(output)}
        write_prepared(name, source_spec(base_source, name=name, label=label, path=output), output, hashes, selected)
        print(json.dumps({"category": category, "output": str(output), "sha256": hashes["output"],
                          "trajectories": len(subset), "frames": FRAMES}), flush=True)
        del dataset, subset
        gc.collect()


if __name__ == "__main__":
    main()
