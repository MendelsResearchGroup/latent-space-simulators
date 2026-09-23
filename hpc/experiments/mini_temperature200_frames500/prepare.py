#!/usr/bin/env python3
"""Audit MetaForge's 200-MD-step family and cache exact 500-frame prefixes."""
from __future__ import annotations

import hashlib
import json
import os
import resource
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/mini_temperature200_frames500"
CACHE = OUT / "data"
BASE = ROOT.parent / "MetaForge/results/mini_temperature_200_1500frames"
FRAMES = 500
SOURCES = {
    "node_low": ("low", "mini_node_optimized_200_500"),
    "node_mixed": ("mixed", "mini_node_optimized_200_500"),
    "stiffness_low": ("low", "mini_stiff_optimized_200_1500"),
    "stiffness_mixed": ("mixed", "mini_stiff_optimized_200_1500"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_digest(graph) -> str:
    edge_index = graph.edge_index.detach().cpu().contiguous()
    nodes = int(getattr(graph, "num_nodes", graph.x.shape[0]))
    return hashlib.sha256(
        f"nodes={nodes};shape={tuple(edge_index.shape)};dtype={edge_index.dtype};".encode()
        + edge_index.numpy().tobytes()
    ).hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def frame_position(graph) -> torch.Tensor:
    position = getattr(graph, "pos", None)
    return position if isinstance(position, torch.Tensor) else graph.x[:, :2]


def audit_source(name: str, temperature: str, family: str) -> tuple[dict, list[str]]:
    folder = BASE / temperature / family
    manifest_path, recipe_path = folder / "manifest.json", folder / "recipe.json"
    manifest, recipe = json.loads(manifest_path.read_text()), json.loads(recipe_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["source_trajectory_count"] == 200
    assert manifest["frames_per_trajectory"] == 1500
    assert manifest["md_steps_between_frames"] == 200
    assert manifest["last_frame_md_step"] == 299800
    assert recipe["config"]["sample_every"] == 200
    assert abs(recipe["strain_increment_per_frame"] - 2e-5) < 1e-14
    assert all(row["source_index"] == index for index, row in enumerate(manifest["records"]))

    selected, records, topologies = [], [], []
    temperatures = Counter()
    for index, record in enumerate(manifest["records"][:120]):
        path = Path(record["trajectory"])
        actual_sha = sha256(path)
        assert actual_sha == record["sha256"], (name, index, actual_sha, record["sha256"])
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        assert isinstance(trajectory, list) and len(trajectory) == 1500
        prefix = trajectory[:FRAMES]
        topology = topology_digest(prefix[0])
        edge_index_sha = tensor_digest(prefix[0].edge_index)
        # These are raw MetaForge graphs.  The repository loader constructs
        # reference_context_positions/ref_edge_attr from frame zero later.
        # Here we verify the raw topology and the stored stiffness channel.
        stiffness_sha = tensor_digest(prefix[0].edge_attr[:, -1])
        for frame, graph in enumerate(prefix):
            assert int(graph.time) == frame * 200, (name, index, frame, graph.time)
            assert torch.isfinite(frame_position(graph)).all(), (name, index, frame)
            assert topology_digest(graph) == topology, (name, index, frame, "topology")
            assert tensor_digest(graph.edge_index) == edge_index_sha, (name, index, frame, "edge_index")
            assert tensor_digest(graph.edge_attr[:, -1]) == stiffness_sha, (name, index, frame, "stiffness")
            graph.source_sim_id = index
        selected.append(prefix)
        topologies.append(topology)
        temperatures[str(record["temperature_K"])] += 1
        records.append({
            "source_index": index,
            "source_chunk": record["source_chunk"],
            "temperature_K": record["temperature_K"],
            "thermal_seed": record["thermal_seed"],
            "trajectory_path": str(path),
            "trajectory_sha256": actual_sha,
            "frame0_topology_sha256": topology,
        })
        del trajectory

    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{name}_ids000_119_frames000_499.pt"
    torch.save(selected, cache)
    del selected
    return ({
        "cache_path": str(cache),
        "cache_sha256": sha256(cache),
        "cache_bytes": cache.stat().st_size,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "recipe_path": str(recipe_path),
        "recipe_sha256": sha256(recipe_path),
        "declared_monolithic_dataset_path": str(folder / "dataset.pt"),
        "declared_monolithic_dataset_sha256": manifest["output_sha256"],
        "selected_temperature_counts": dict(temperatures),
        "records": records,
    }, topologies)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    datasets, topology = {}, {}
    for name, (temperature, family) in SOURCES.items():
        datasets[name], topology[name] = audit_source(name, temperature, family)
    for low, mixed in (("node_low", "node_mixed"), ("stiffness_low", "stiffness_mixed")):
        assert topology[low] == topology[mixed]
        low_records, mixed_records = datasets[low]["records"], datasets[mixed]["records"]
        assert all(
            a["source_index"] == b["source_index"]
            and a["source_chunk"] == b["source_chunk"]
            and a["thermal_seed"] == b["thermal_seed"]
            for a, b in zip(low_records, mixed_records)
        )
    assert not set(topology["node_low"]).intersection(topology["stiffness_low"])
    result = {
        "status": "complete",
        "study": "mini_temperature200_frames500",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "source_frames": 1500,
        "cached_prefix_frames": 500,
        "saved_md_step_stride": 200,
        "cached_time_range_md_steps": [0, 99800],
        "cached_total_axial_strain": 0.00998,
        "split": {"train_max": list(range(20)), "validation": list(range(20, 40)), "heldout_development": list(range(40, 120))},
        "cache_mutation": "Only source_sim_id provenance is assigned; raw graph tensors and the first 500 serialized frames are unchanged. The standard repository loader later derives static reference fields from frame zero.",
        "pairing": "All 120 selected low/mixed source_chunk, source_index, thermal_seed and topology identities match within each family.",
        "cross_family_topology_overlap": 0,
        "datasets": datasets,
        "script_sha256": sha256(Path(__file__)),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (OUT / "data_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "cached_prefix_frames", "cached_time_range_md_steps", "pbs_job_id", "peak_rss_kib")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
