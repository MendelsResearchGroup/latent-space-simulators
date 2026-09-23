"""Export response-blind per-edge latent sensitivity for one fixed held-out graph/source."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
for source_root in (ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.simulation import batch_delta_graphs, compact_reference_edge_features, reference_positions_for_model


BASE = ROOT / "notebooks/results/all4_coverage_1000"
OUT = ROOT / "notebooks/results/edge_attribution_one_vs_twenty"
MODELS = {
    "1 per source": BASE / "all4_one_each_retry_v5_validation30",
    "20 per source": BASE / "all4_twenty_each_retry_v5_validation30",
}
SOURCES = ("node_low", "stiffness_low")
TRAJECTORY_ID = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(result, manifest, source_name: str):
    entry = manifest["datasets"][source_name]
    spec = {
        "name": source_name,
        "label": source_name,
        "path": entry["test_path"],
        "train_count": 0,
        "val_count": 1,
        "split_indices": {"train": [], "val": [TRAJECTORY_ID], "test": []},
    }
    source = {
        "name": source_name,
        "label": source_name,
        "path": entry["test_path"],
        "dataset_name": source_name,
        "source_name": source_name,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [spec],
    }
    _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    if len(simulations) != 1:
        raise AssertionError(f"Expected one graph for {source_name}, got {len(simulations)}")
    return simulations[0]


def normalized_inputs(result, simulation):
    params, norm = result["params"], result["normalizers"]
    batch = batch_delta_graphs(
        [simulation], [(0, 0)], pos_dim=2, device="cpu",
        node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"],
    )
    return {
        "node": (batch["node_feature"] - norm["node_feature_mean"]) / norm["node_feature_std"],
        "ref_pos": batch["ref_pos"],
        "edge": ((batch["edge_attr"] - norm["edge_mean"]) / norm["edge_std"]).detach().requires_grad_(True),
        "ref_edge": ((batch["ref_edge_attr"] - norm.get("ref_edge_mean", norm["edge_mean"])) /
                     norm.get("ref_edge_std", norm["edge_std"])).detach().requires_grad_(True),
        "edge_index": batch["edge_index"],
        "batch": batch["batch"],
    }


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    rows = []
    checkpoint_hashes = {}
    first_dir = next(iter(MODELS.values()))
    first_recipe = json.loads((first_dir / "propagator_recipe.json").read_text())
    first_result = load_experiment_bundle(first_dir / "propagator.pt", cfg=first_recipe["config"], device="cpu")
    source_data = {source: load_graph(first_result, manifest, source) for source in SOURCES}
    for training, model_dir in MODELS.items():
        if model_dir == first_dir:
            result = first_result
        else:
            recipe = json.loads((model_dir / "propagator_recipe.json").read_text())
            result = load_experiment_bundle(model_dir / "propagator.pt", cfg=recipe["config"], device="cpu")
        model = result["ae"].eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint_hashes[training] = sha256(model_dir / "ae.pt")
        for source in SOURCES:
            simulation = source_data[source]
            inputs = normalized_inputs(result, simulation)
            latent, _ = model.encode(inputs["node"], inputs["ref_pos"], inputs["edge"],
                                     inputs["ref_edge"], inputs["edge_index"], inputs["batch"])
            gradients = []
            for coordinate in range(2):
                gradients.append(torch.autograd.grad(
                    latent[0, coordinate], (inputs["edge"], inputs["ref_edge"]),
                    retain_graph=coordinate == 0,
                ))
            edge_contrib = torch.stack([gradient[0] * inputs["edge"].detach() for gradient in gradients])
            ref_contrib = torch.stack([gradient[1] * inputs["ref_edge"].detach() for gradient in gradients])
            geometry_score = torch.sqrt(
                edge_contrib[:, :, :3].square().sum((0, 2)) + ref_contrib[:, :, :3].square().sum((0, 2))
            )
            stiffness_score = torch.sqrt(
                edge_contrib[:, :, 3].square().sum(0) + ref_contrib[:, :, 3].square().sum(0)
            )
            total_score = torch.sqrt(geometry_score.square() + stiffness_score.square())

            graph = simulation[0]
            position = reference_positions_for_model(graph, pos_dim=2, device="cpu").numpy()
            raw_edge = compact_reference_edge_features(graph, pos_dim=2, device="cpu")
            edge_index = graph.edge_index.detach().cpu().numpy()
            groups = {}
            for edge_id, (left, right) in enumerate(edge_index.T):
                key = tuple(sorted((int(left), int(right))))
                groups.setdefault(key, []).append(edge_id)
            for edge_id, ((left, right), directed) in enumerate(sorted(groups.items())):
                directed_index = torch.tensor(directed, dtype=torch.long)
                rows.append({
                    "training": training,
                    "source": source,
                    "trajectory_id": TRAJECTORY_ID,
                    "edge_id": edge_id,
                    "node_i": left,
                    "node_j": right,
                    "x_i": float(position[left, 0]),
                    "y_i": float(position[left, 1]),
                    "x_j": float(position[right, 0]),
                    "y_j": float(position[right, 1]),
                    "length": float(raw_edge[directed_index, 2].mean()),
                    "stiffness": float(raw_edge[directed_index, 3].mean()),
                    "geometry_influence": float(torch.linalg.vector_norm(geometry_score[directed_index])),
                    "stiffness_influence": float(torch.linalg.vector_norm(stiffness_score[directed_index])),
                    "total_influence": float(torch.linalg.vector_norm(total_score[directed_index])),
                    "z0": float(latent[0, 0].detach()),
                    "z1": float(latent[0, 1].detach()),
                    "directed_entries": len(directed),
                })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "edge_influence.csv", index=False)
    metadata = {
        "status": "complete",
        "sources": list(SOURCES),
        "trajectory_id": TRAJECTORY_ID,
        "frame": 0,
        "models": {name: str(path.relative_to(ROOT)) for name, path in MODELS.items()},
        "checkpoint_sha256": checkpoint_hashes,
        "metric": "paired-undirected-edge L2 norm across z0/z1 of gradient times normalized input; current and reference edge routes combined",
        "geometry_channels": "normalized edge vector x/y and length",
        "stiffness_channel": "normalized stored stiffness",
        "qualification": "local response-blind sensitivity, not a causal deletion score; magnitudes are normalized within each panel for visualization",
        "script_sha256": sha256(Path(__file__)),
        "data_manifest_sha256": sha256(BASE / "data_manifest.json"),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(table.groupby(["training", "source"])[["geometry_influence", "stiffness_influence", "total_influence"]].agg(["count", "mean", "max"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
