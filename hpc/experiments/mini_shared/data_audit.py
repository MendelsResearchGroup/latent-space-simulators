#!/usr/bin/env python3
"""Audit and prepare the fixed 200-network mini shared-study cohorts.

This intentionally loads one original/mixed-temperature pair at a time on a
shared PBS node.  It never normalizes or otherwise mutates graph objects:
the saved train/validation and test caches are direct trajectory subsets.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/mini_shared"
CACHE = OUT / "data"
EXTERNAL = ROOT.parent / "MetaForge/results/mini_mixed_temperature"
PYTHON = Path(__file__)

SOURCES = {
    "node": {
        "original": ROOT.parent / "data/mini_node_optimized_200_200.pt",
        "mixed": ROOT.parent / "data/mini_node_optimized_mixed_temp_200_200.pt",
        "metaforge": EXTERNAL / "mini_node_optimized_200_500",
    },
    "stiffness": {
        "original": ROOT.parent / "data/mini_stiff_optimized_200_200.pt",
        "mixed": ROOT.parent / "data/mini_stiff_optimized_mixed_temp_200_200.pt",
        "metaforge": EXTERNAL / "mini_stiff_optimized_200_1500",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def graph_fields(graph) -> dict:
    if hasattr(graph, "keys"):
        keys = list(graph.keys())
    else:
        keys = sorted(vars(graph))
    fields = {}
    for key in sorted(keys):
        value = getattr(graph, key)
        if isinstance(value, torch.Tensor):
            fields[key] = {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            fields[key] = {"kind": type(value).__name__, "value": scalar(value)}
    return fields


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def topology_digest(graph) -> str:
    if not hasattr(graph, "edge_index"):
        raise KeyError("Graph has no edge_index.")
    node_count = int(getattr(graph, "num_nodes", graph.x.shape[0]))
    edge_index = graph.edge_index
    return hashlib.sha256(
        f"nodes={node_count};shape={tuple(edge_index.shape)};dtype={edge_index.dtype};".encode()
        + edge_index.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def coordinate_tensor(graph) -> torch.Tensor:
    if hasattr(graph, "pos") and isinstance(graph.pos, torch.Tensor):
        return graph.pos
    if hasattr(graph, "x") and isinstance(graph.x, torch.Tensor):
        return graph.x[:, :2]
    raise KeyError("Graph has neither tensor pos nor tensor x coordinates.")


def identity(graph) -> dict:
    names = ("source_chunk", "source_sim_id", "source_frame_index", "sim_id")
    return {name: scalar(getattr(graph, name)) for name in names if hasattr(graph, name)}


def audit_pair(name: str, spec: dict) -> dict:
    original_path, mixed_path = spec["original"], spec["mixed"]
    mf_manifest = json.loads((spec["metaforge"] / "manifest.json").read_text())
    mf_recipe = json.loads((spec["metaforge"] / "recipe.json").read_text())
    expected_records = mf_manifest["records"]
    if len(expected_records) != 200:
        raise ValueError(f"{name}: expected 200 MetaForge records.")
    start = time.monotonic()
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    mixed = torch.load(mixed_path, map_location="cpu", weights_only=False)
    load_seconds = time.monotonic() - start
    if not isinstance(original, list) or not isinstance(mixed, list):
        raise TypeError(f"{name}: expected top-level trajectory lists.")
    if len(original) != 200 or len(mixed) != 200:
        raise ValueError(f"{name}: expected exactly 200 original and mixed trajectories.")

    finite = {"original": 0, "mixed": 0}
    frame_counts = {"original": Counter(), "mixed": Counter()}
    correspondence = []
    source_ids = []
    schema = None
    for index, (orig_sim, mixed_sim, record) in enumerate(zip(original, mixed, expected_records)):
        if len(orig_sim) != 200 or len(mixed_sim) != 200:
            raise ValueError(f"{name} index {index}: expected 200 frames in both paired trajectories.")
        frame_counts["original"][len(orig_sim)] += 1
        frame_counts["mixed"][len(mixed_sim)] += 1
        if schema is None:
            schema = {"original_frame0": graph_fields(orig_sim[0]), "mixed_frame0": graph_fields(mixed_sim[0])}
        pair_ok = True
        topo = []
        for frame, (orig_graph, mixed_graph) in enumerate(zip(orig_sim, mixed_sim)):
            orig_xy, mixed_xy = coordinate_tensor(orig_graph), coordinate_tensor(mixed_graph)
            if not bool(torch.isfinite(orig_xy).all()):
                raise ValueError(f"{name} original index {index} frame {frame}: non-finite coordinates.")
            if not bool(torch.isfinite(mixed_xy).all()):
                raise ValueError(f"{name} mixed index {index} frame {frame}: non-finite coordinates.")
            finite["original"] += orig_xy.numel()
            finite["mixed"] += mixed_xy.numel()
            orig_topology, mixed_topology = topology_digest(orig_graph), topology_digest(mixed_graph)
            if orig_topology != mixed_topology:
                pair_ok = False
            topo.append(orig_topology)
        source_identity = identity(orig_sim[0])
        mixed_identity = identity(mixed_sim[0])
        # The converted full-200 originals need not carry MetaForge's
        # source_chunk attribute.  Their paired identity is instead checked
        # from every topology plus every common serialized identity field;
        # source_chunk remains the external generation record.
        common_identity = sorted(set(source_identity).intersection(mixed_identity))
        identity_matches = all(source_identity[key] == mixed_identity[key] for key in common_identity)
        if not identity_matches:
            raise ValueError(f"{name} index {index}: paired serialized identity fields differ.")
        if not pair_ok:
            raise ValueError(f"{name} index {index}: original/mixed topology mismatch.")
        correspondence.append({
            "source_index": index,
            "metaforge_source_chunk": record["source_chunk"],
            "temperature_K": record["temperature_K"],
            "original_identity": source_identity,
            "mixed_identity": mixed_identity,
            "common_identity_fields": common_identity,
            "common_identity_fields_match": identity_matches,
            "frame0_topology_sha256": topo[0],
            "topology_constant_over_200_frames": len(set(topo)) == 1,
        })
        source_ids.append(source_identity)

    if not all(row["topology_constant_over_200_frames"] for row in correspondence):
        raise ValueError(f"{name}: topology changes within one or more trajectories.")
    # Direct list slicing preserves the serialized graph objects and their source IDs.
    for variant, data in (("original", original), ("mixed_temp", mixed)):
        for split, subset in (("trainval40", data[:40]), ("test80", data[40:120])):
            output = CACHE / f"mini_{name}_{variant}_{split}_200frames.pt"
            torch.save(subset, output)
    del original, mixed
    return {
        "files": {
            "original": {"path": str(original_path), "sha256": sha256(original_path)},
            "mixed_temp": {"path": str(mixed_path), "sha256": sha256(mixed_path),
                           "metaforge_output_sha256": mf_manifest["output_sha256"]},
        },
        "exact_shape": {"original": [200, 200], "mixed_temp": [200, 200]},
        "schema_frame0": schema,
        "finite_coordinate_values_checked": finite,
        "paired_original_mixed": {"all_200_topologies_match": True, "records": correspondence},
        "metaforge": {
            "manifest_path": str(spec["metaforge"] / "manifest.json"),
            "recipe_path": str(spec["metaforge"] / "recipe.json"),
            "manifest_sha256": sha256(spec["metaforge"] / "manifest.json"),
            "recipe_sha256": sha256(spec["metaforge"] / "recipe.json"),
            "temperature_counts": mf_manifest["temperature_counts"],
            "source_sha256": mf_recipe["source_sha256"],
            "assignment_seed": mf_recipe["assignment_seed"],
            "thermal_seed_rule": mf_recipe["thermal_seed_rule"],
            "protocol": {key: mf_recipe[key] for key in ("frame_selection", "units", "strain_increment_per_frame", "protocol_basis", "config")},
        },
        "load_seconds": load_seconds,
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    audit = {name: audit_pair(name, spec) for name, spec in SOURCES.items()}
    caches = {}
    for path in sorted(CACHE.glob("*.pt")):
        caches[path.name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    datasets = {}
    for source in ("node", "stiffness"):
        for variant, suffix in (("original", "original"), ("mixed_temp", "mixed")):
            prefix = f"mini_{source}_{variant}"
            trainval = caches[f"{prefix}_trainval40_200frames.pt"]
            test = caches[f"{prefix}_test80_200frames.pt"]
            datasets[f"{source}_{suffix}"] = {
                "trainval_path": trainval["path"], "trainval_sha256": trainval["sha256"],
                "test_path": test["path"], "test_sha256": test["sha256"],
            }
    node_topologies = {row["frame0_topology_sha256"] for row in audit["node"]["paired_original_mixed"]["records"][:120]}
    stiffness_topologies = {row["frame0_topology_sha256"] for row in audit["stiffness"]["paired_original_mixed"]["records"][:120]}
    manifest = {
        "status": "complete",
        "purpose": "Data audit and unmodified serialized 200-frame caches; no training or mixed-temperature training decision.",
        "audit_script": {"path": str(PYTHON), "sha256": sha256(PYTHON)},
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "split": {
            "deterministic_seed": 123,
            "cohort": "first 120 serialized source indices; paired original/mixed-temperature entries share every split",
            "train": list(range(0, 20)), "validation": list(range(20, 40)),
            "test": list(range(40, 120)), "unused": list(range(120, 200)),
            "cache_mapping": "trainval40 preserves source indices 0..39; test80 preserves original source_sim_id attributes, while a loader that uses list indices must add 40 for split-local indices.",
        },
        "graph_mutation": "none; caches are direct list slices of torch.load objects, without compatibility conversion.",
        "datasets": datasets,
        "cross_branch_topology_overlap_first120": {
            "node_unique": len(node_topologies), "stiffness_unique": len(stiffness_topologies),
            "intersection_count": len(node_topologies.intersection(stiffness_topologies)),
            "intersection_sha256": sorted(node_topologies.intersection(stiffness_topologies)),
        },
        "sources": audit,
        "caches": caches,
    }
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
