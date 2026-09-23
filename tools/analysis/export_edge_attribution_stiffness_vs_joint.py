"""Compare stiffness-edge latent sensitivity for stiffness-only and joint-original AEs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
for source_root in (ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.simulation import (
    batch_delta_graphs,
    compact_reference_edge_features,
    reference_positions_for_model,
)


BASE = ROOT / "notebooks/results/mini_shared"
OUT = ROOT / "notebooks/results/edge_attribution_stiffness_vs_joint"
MODELS = {
    "20 stiffness only": BASE / "stiffness_only",
    "20 stiffness + 20 node": BASE / "joint_original",
}
SOURCE = "stiffness_original"
TRAJECTORY_ID = 40
TEST_LOCAL_INDEX = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_heldout_graph(result: dict, manifest: dict):
    path = manifest["datasets"][SOURCE]["test_path"]
    component = {
        "name": SOURCE,
        "label": SOURCE,
        "path": path,
        "train_count": 0,
        "val_count": 1,
        "split_indices": {"train": [], "val": [TEST_LOCAL_INDEX], "test": []},
    }
    source = {
        "name": SOURCE,
        "label": SOURCE,
        "path": path,
        "dataset_name": SOURCE,
        "source_name": SOURCE,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [component],
    }
    _, simulations, _, _ = resolve_train_val_test(
        source, result["params"], split_seed=123
    )
    if len(simulations) != 1:
        raise AssertionError(f"Expected one held-out graph, found {len(simulations)}.")
    return simulations[0]


def normalized_inputs(result: dict, simulation, frame: int):
    params, norm = result["params"], result["normalizers"]
    packed = batch_delta_graphs(
        [simulation],
        [(0, frame)],
        pos_dim=2,
        device="cpu",
        node_feature_mode=params["node_feature_mode"],
        edge_mode=params["edge_mode"],
    )
    return {
        "node": ((
            packed["node_feature"] - norm["node_feature_mean"]
        ) / norm["node_feature_std"]).detach().requires_grad_(True),
        "ref_pos": packed["ref_pos"].detach().requires_grad_(True),
        "edge": (
            (packed["edge_attr"] - norm["edge_mean"]) / norm["edge_std"]
        ).detach().requires_grad_(True),
        "ref_edge": (
            (packed["ref_edge_attr"] - norm.get("ref_edge_mean", norm["edge_mean"]))
            / norm.get("ref_edge_std", norm["edge_std"])
        ).detach().requires_grad_(True),
        "edge_index": packed["edge_index"],
        "batch": packed["batch"],
    }


def latent_gradients(model, inputs: dict):
    latent, _ = model.encode(
        inputs["node"], inputs["ref_pos"], inputs["edge"], inputs["ref_edge"],
        inputs["edge_index"], inputs["batch"],
    )
    differentiable = (
        inputs["node"], inputs["ref_pos"], inputs["edge"], inputs["ref_edge"]
    )
    gradients = [
        torch.autograd.grad(
            latent[0, coordinate], differentiable, retain_graph=coordinate == 0
        )
        for coordinate in range(2)
    ]
    return latent, [torch.stack([gradient[index] for gradient in gradients]) for index in range(4)]


def l2_score(gradient: torch.Tensor, displacement: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(gradient * displacement.unsqueeze(0)))


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = BASE / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    first_dir = next(iter(MODELS.values()))
    first_recipe = json.loads((first_dir / "propagator_recipe.json").read_text())
    first_result = load_experiment_bundle(
        first_dir / "propagator.pt", cfg=first_recipe["config"], device="cpu"
    )
    simulation = load_heldout_graph(first_result, manifest)
    graph = simulation[0]
    position = reference_positions_for_model(graph, pos_dim=2, device="cpu")
    raw_edge = compact_reference_edge_features(graph, pos_dim=2, device="cpu")
    edge_index = graph.edge_index.detach().cpu()

    groups: dict[tuple[int, int], list[int]] = {}
    for stored_id, (left, right) in enumerate(edge_index.T.tolist()):
        groups.setdefault(tuple(sorted((int(left), int(right)))), []).append(stored_id)

    rows = []
    feature_rows = []
    checkpoint_hashes = {}
    for label, model_dir in MODELS.items():
        if model_dir == first_dir:
            result = first_result
        else:
            recipe = json.loads((model_dir / "propagator_recipe.json").read_text())
            result = load_experiment_bundle(
                model_dir / "propagator.pt", cfg=recipe["config"], device="cpu"
            )
        model = result["ae"].eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checkpoint_hashes[label] = sha256(model_dir / "ae.pt")

        baseline = normalized_inputs(result, simulation, 0)
        baseline_values = {
            name: baseline[name].detach() for name in ("node", "edge")
        }
        frame_results = {}
        for frame in (0, 1, 2, 3, 4, 5, 20):
            inputs = normalized_inputs(result, simulation, frame)
            latent, gradients = latent_gradients(model, inputs)
            node_gradient, ref_pos_gradient, edge_gradient, ref_edge_gradient = gradients
            ref_position = inputs["ref_pos"].detach()
            ref_position = ref_position - ref_position.mean(dim=0, keepdim=True)
            feature_scores = {
                "reference node positions": l2_score(ref_pos_gradient, ref_position),
                "reference edge directions": l2_score(ref_edge_gradient[:, :, :2], inputs["ref_edge"].detach()[:, :2]),
                "reference edge lengths": l2_score(ref_edge_gradient[:, :, 2:3], inputs["ref_edge"].detach()[:, 2:3]),
                "spring stiffnesses": l2_score(ref_edge_gradient[:, :, 3:4], inputs["ref_edge"].detach()[:, 3:4]),
                "node displacement": l2_score(node_gradient, inputs["node"].detach()-baseline_values["node"]),
                "edge direction change": l2_score(edge_gradient[:, :, :2], inputs["edge"].detach()[:, :2]-baseline_values["edge"][:, :2]),
                "edge length change": l2_score(edge_gradient[:, :, 2:3], inputs["edge"].detach()[:, 2:3]-baseline_values["edge"][:, 2:3]),
                "relative edge stretch": l2_score(edge_gradient[:, :, 3:4], inputs["edge"].detach()[:, 3:4]-baseline_values["edge"][:, 3:4]),
            }
            total = sum(feature_scores.values())
            for feature, score in feature_scores.items():
                feature_rows.append({
                    "training": label,
                    "source": SOURCE,
                    "trajectory_id": TRAJECTORY_ID,
                    "frame": frame,
                    "feature": feature,
                    "influence": score,
                    "influence_fraction": score / total if total > 0 else float("nan"),
                })
            frame_results[frame] = (latent.detach(), inputs, ref_edge_gradient)

        latent, inputs, ref_edge_gradient = frame_results[0]
        stiffness_score = torch.sqrt(
            (ref_edge_gradient[:, :, 3] * inputs["ref_edge"].detach()[:, 3]).square().sum(0)
        )

        for edge_id, ((left, right), stored_ids) in enumerate(sorted(groups.items())):
            indices = torch.tensor(stored_ids, dtype=torch.long)
            rows.append({
                "training": label,
                "source": SOURCE,
                "trajectory_id": TRAJECTORY_ID,
                "edge_id": edge_id,
                "node_i": left,
                "node_j": right,
                "x_i": float(position[left, 0]),
                "y_i": float(position[left, 1]),
                "x_j": float(position[right, 0]),
                "y_j": float(position[right, 1]),
                "length": float(raw_edge[indices, 2].mean()),
                "stiffness": float(raw_edge[indices, 3].mean()),
                "stiffness_influence": float(torch.linalg.vector_norm(stiffness_score[indices])),
                "z0": float(latent[0, 0].detach()),
                "z1": float(latent[0, 1].detach()),
                "stored_entries": len(stored_ids),
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "edge_influence.csv", index=False)
    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(OUT / "feature_influence.csv", index=False)
    x_low, y_low = position.min(dim=0).values.tolist()
    x_high, y_high = position.max(dim=0).values.tolist()
    metadata = {
        "status": "complete",
        "source": SOURCE,
        "trajectory_id": TRAJECTORY_ID,
        "test_local_index": TEST_LOCAL_INDEX,
        "frame": 0,
        "models": {name: str(path.relative_to(ROOT)) for name, path in MODELS.items()},
        "checkpoint_sha256": checkpoint_hashes,
        "test_data_sha256": manifest["datasets"][SOURCE]["test_sha256"],
        "metric": "gradient times input, L2 aggregated across z0/z1; static inputs measured relative to fitted zero/center and dynamic inputs relative to frame 0",
        "edge_metric": "per-undirected-edge L2 norm across z0/z1 of gradient times normalized reference-stiffness input",
        "feature_frames": [0, 1, 2, 3, 4, 5, 20],
        "qualification": "local response-blind sensitivity, not a causal deletion score; feature families differ in element count and are descriptive",
        "drawing_box": {"x1": x_low, "x2": x_high, "y1": y_low, "y2": y_high},
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(feature_table.sort_values(["training", "frame", "influence_fraction"], ascending=[True, True, False]).to_string(index=False))
    print(table.groupby("training")["stiffness_influence"].agg(["count", "mean", "max"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
