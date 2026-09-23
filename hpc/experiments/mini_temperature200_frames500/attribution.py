#!/usr/bin/env python3
"""Response-blind feature and edge attribution across four frozen 500-frame AEs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.simulation import batch_delta_graphs, compact_reference_edge_features, reference_positions_for_model


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/mini_temperature200_frames500"
OUT = BASE / "edge_attribution"
MODELS = {
    "1 per each of four sources": BASE / "all4_one_each",
    "20 per each of four sources": BASE / "all4_twenty_each",
    "20 stiffness low-T only": BASE / "stiffness_low_only",
    "20 stiffness + 20 node low-T": BASE / "joint_low",
}
SOURCE = "stiffness_low"
TRAJECTORY_ID = 40
FRAMES = (0, 1, 2, 3, 4, 5, 20, 50)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heldout(result: dict, manifest: dict):
    path = manifest["datasets"][SOURCE]["cache_path"]
    component = {"name": SOURCE, "label": SOURCE, "path": path, "train_count": 0, "val_count": 1, "split_indices": {"train": [], "val": [TRAJECTORY_ID], "test": []}}
    source = {"name": SOURCE, "label": SOURCE, "path": path, "dataset_name": SOURCE, "source_name": SOURCE, "coordinate_normalization": "position_normalization", "dataset_mixture": [component]}
    _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    assert len(simulations) == 1 and int(simulations[0][0].source_sim_id) == TRAJECTORY_ID
    return simulations[0]


def normalized_inputs(result: dict, simulation, frame: int) -> dict:
    params, norm = result["params"], result["normalizers"]
    packed = batch_delta_graphs([simulation], [(0, frame)], pos_dim=2, device="cpu", node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"])
    return {
        "node": ((packed["node_feature"] - norm["node_feature_mean"]) / norm["node_feature_std"]).detach().requires_grad_(True),
        "ref_pos": packed["ref_pos"].detach().requires_grad_(True),
        "edge": ((packed["edge_attr"] - norm["edge_mean"]) / norm["edge_std"]).detach().requires_grad_(True),
        "ref_edge": ((packed["ref_edge_attr"] - norm.get("ref_edge_mean", norm["edge_mean"])) / norm.get("ref_edge_std", norm["edge_std"])).detach().requires_grad_(True),
        "edge_index": packed["edge_index"], "batch": packed["batch"],
    }


def latent_gradients(model, inputs: dict):
    latent, _ = model.encode(inputs["node"], inputs["ref_pos"], inputs["edge"], inputs["ref_edge"], inputs["edge_index"], inputs["batch"])
    values = (inputs["node"], inputs["ref_pos"], inputs["edge"], inputs["ref_edge"])
    gradients = [torch.autograd.grad(latent[0, coordinate], values, retain_graph=coordinate == 0) for coordinate in range(2)]
    return latent, [torch.stack([gradient[index] for gradient in gradients]) for index in range(4)]


def score(gradient: torch.Tensor, value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(gradient * value.unsqueeze(0)))


def main() -> None:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    feature_rows, edge_rows, hashes = [], [], {}
    simulation = None
    groups = None
    raw_reference_edge = None
    position = None
    for label, folder in MODELS.items():
        recipe = json.loads((folder / "propagator_recipe.json").read_text())
        result = load_experiment_bundle(folder / "propagator.pt", cfg=recipe["config"], device="cpu")
        current_simulation = heldout(result, manifest)
        if simulation is None:
            simulation = current_simulation
            graph = simulation[0]
            position = reference_positions_for_model(graph, pos_dim=2, device="cpu")
            raw_reference_edge = compact_reference_edge_features(graph, pos_dim=2, device="cpu")
            groups = {}
            for stored_id, (left, right) in enumerate(graph.edge_index.detach().cpu().T.tolist()):
                groups.setdefault(tuple(sorted((int(left), int(right)))), []).append(stored_id)
        hashes[label] = sha(folder / "ae.pt")
        model = result["ae"].eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        baseline = normalized_inputs(result, current_simulation, 0)
        baseline_node, baseline_edge = baseline["node"].detach(), baseline["edge"].detach()
        for frame in FRAMES:
            inputs = normalized_inputs(result, current_simulation, frame)
            latent, gradients = latent_gradients(model, inputs)
            node_gradient, ref_position_gradient, edge_gradient, ref_edge_gradient = gradients
            centered_reference = inputs["ref_pos"].detach() - inputs["ref_pos"].detach().mean(0, keepdim=True)
            values = {
                "reference positions": score(ref_position_gradient, centered_reference),
                "reference edge direction": score(ref_edge_gradient[:, :, :2], inputs["ref_edge"].detach()[:, :2]),
                "reference edge length": score(ref_edge_gradient[:, :, 2:3], inputs["ref_edge"].detach()[:, 2:3]),
                "spring stiffness": score(ref_edge_gradient[:, :, 3:4], inputs["ref_edge"].detach()[:, 3:4]),
                "node displacement from frame 0": score(node_gradient, inputs["node"].detach() - baseline_node),
                "edge direction change from frame 0": score(edge_gradient[:, :, :2], inputs["edge"].detach()[:, :2] - baseline_edge[:, :2]),
                "edge length change from frame 0": score(edge_gradient[:, :, 2:3], inputs["edge"].detach()[:, 2:3] - baseline_edge[:, 2:3]),
                "relative edge stretch from frame 0": score(edge_gradient[:, :, 3:4], inputs["edge"].detach()[:, 3:4] - baseline_edge[:, 3:4]),
            }
            total = sum(values.values())
            for feature, influence in values.items():
                feature_rows.append({"training": label, "source": SOURCE, "trajectory_id": TRAJECTORY_ID, "frame": frame, "md_step": frame * 200, "feature": feature, "influence": influence, "influence_fraction": influence / total if total else float("nan")})
            if frame == 0:
                geometry = torch.sqrt((ref_edge_gradient[:, :, :3] * inputs["ref_edge"].detach()[:, :3].unsqueeze(0)).square().sum((0, 2)))
                stiffness = torch.sqrt((ref_edge_gradient[:, :, 3] * inputs["ref_edge"].detach()[:, 3].unsqueeze(0)).square().sum(0))
                for edge_id, ((left, right), stored_ids) in enumerate(sorted(groups.items())):
                    indices = torch.tensor(stored_ids, dtype=torch.long)
                    edge_rows.append({
                        "training": label, "source": SOURCE, "trajectory_id": TRAJECTORY_ID, "edge_id": edge_id,
                        "node_i": left, "node_j": right, "x_i": float(position[left, 0]), "y_i": float(position[left, 1]),
                        "x_j": float(position[right, 0]), "y_j": float(position[right, 1]),
                        "length": float(raw_reference_edge[indices, 2].mean()), "stiffness": float(raw_reference_edge[indices, 3].mean()),
                        "reference_edge_geometry_influence": float(torch.linalg.vector_norm(geometry[indices])),
                        "stiffness_influence": float(torch.linalg.vector_norm(stiffness[indices])), "stored_entries": len(stored_ids),
                        "z0": float(latent[0, 0]), "z1": float(latent[0, 1]),
                    })
    pd.DataFrame(feature_rows).to_csv(OUT / "feature_influence.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(OUT / "edge_influence.csv", index=False)
    metadata = {
        "status": "complete", "source": SOURCE, "trajectory_id": TRAJECTORY_ID,
        "models": {label: str(path.relative_to(ROOT)) for label, path in MODELS.items()},
        "checkpoint_sha256": hashes, "data_manifest_sha256": sha(BASE / "data_manifest.json"),
        "feature_frames": list(FRAMES), "metric": "gradient times normalized input, L2 across both raw latent coordinates",
        "dynamic_baseline": "per-feature normalized value at frame0; includes node displacement and edge direction/length/stretch changes",
        "edge_metric": "paired directed entries aggregated to physical undirected springs",
        "qualification": "Local response-blind sensitivity on one fixed heldout graph, not causal deletion and not comparable as absolute mass across unequal-size feature families.",
        "script_sha256": sha(Path(__file__)), "job_id": os.environ.get("PBS_JOBID"),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
