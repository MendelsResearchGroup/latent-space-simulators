#!/usr/bin/env python3
"""Compare physical, random-augmented, and frozen-latent-augmented LJ GNNs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
CODE = Path(os.environ["LSS_CODE_ROOT"]).resolve()
BROAD_CODE = ROOT / "notebooks/results/lj100_broad/code_v1"
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(BROAD_CODE / "src"))

from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from lss.dynamics.experiment import _autoencoder_class
from lss.dynamics.trajectory_propagator import make_trajectory_propagator
from lss.graph import clone_graph

_influence_spec = importlib.util.spec_from_file_location(
    "latent_influence_gnn", CODE / "src/lss/dynamics/latent_influence_gnn.py"
)
assert _influence_spec is not None and _influence_spec.loader is not None
_influence = importlib.util.module_from_spec(_influence_spec)
_influence_spec.loader.exec_module(_influence)
NormalizedAccelerationGNN = _influence.NormalizedAccelerationGNN
canonical_undirected = _influence.canonical_undirected
geometric_edge_features = _influence.geometric_edge_features
random_nonphysical_edges = _influence.random_nonphysical_edges
select_topk_nonphysical_edges = _influence.select_topk_nonphysical_edges


OUT = ROOT / "notebooks/results/lj_latent_influence_edges"
HORIZONS = (50, 100)
INFLUENCE_HORIZONS = (5, 10, 20)
VALIDATION_FRAMES = (10, 25, 50, 75, 99)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_ae(data: dict):
    bundle = torch.load(data["ae_checkpoint"], map_location="cpu", weights_only=False)
    params, norm = bundle["params"], data["normalizers"]
    ae = _autoencoder_class(params["autoencoder_model"])(
        pos_dim=2,
        node_feature_dim=norm["node_feature_mean"].numel(),
        edge_dim=norm["edge_mean"].numel(),
        hidden_size=params["hidden_size"],
        latent_dim=2,
        latent_tokens=params["latent_tokens"],
        reconstruction_dim=norm["target_mean"].numel(),
        message_passing_steps=params["message_passing_steps"],
    )
    ae.set_edge_normalization(
        norm["edge_mean"], norm["edge_std"], norm["ref_edge_mean"], norm["ref_edge_std"]
    )
    ae.edge_mode = params["edge_mode"]
    ae.load_state_dict(bundle["ae_state_dict"], strict=True)
    ae.eval()
    for parameter in ae.parameters():
        parameter.requires_grad_(False)
    return ae


def decode_normalized_positions(ae, row: dict, z: torch.Tensor, data: dict) -> torch.Tensor:
    batch = torch.zeros(len(row["h_ref"]), dtype=torch.long)
    stats, norm = data["latent_stats"], data["normalizers"]
    raw_z = z.reshape(1, -1) * stats["z_std"] + stats["z_mean"]
    delta = ae.decode(raw_z, row["h_ref"], batch)
    return row["ref_pos"] + (delta * norm["target_std"] + norm["target_mean"]) * row["span"]


def latent_score(ae, latent_model, row: dict, data: dict) -> torch.Tensor:
    """Shared-mode Gram matrix from centered perturbations of the global latent."""

    initial = row["latent_norm"][0].detach()
    epsilon = 1e-3
    response_columns = []
    with torch.no_grad():
        for coordinate in range(initial.numel()):
            direction = torch.zeros_like(initial)
            direction[coordinate] = epsilon
            positive = latent_model(initial + direction, row["reference_tokens"])
            negative = latent_model(initial - direction, row["reference_tokens"])
            for horizon in INFLUENCE_HORIZONS:
                positive_positions = decode_normalized_positions(ae, row, positive[horizon], data)
                negative_positions = decode_normalized_positions(ae, row, negative[horizon], data)
                response_columns.append(((positive_positions - negative_positions) / (2.0 * epsilon)).abs())
    participation = torch.cat(response_columns, dim=1)
    score = participation @ participation.t()
    scale = score.square().mean().sqrt().clamp_min(1e-12)
    return score / scale


def normalized_positions(row: dict) -> torch.Tensor:
    half = row["half"].float()
    center = row["center"].float()
    return (row["physical_positions"].float() - center) / half


def base_relation(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    graph = row["physical_reference"]
    base = canonical_undirected(graph.edge_index)
    node_count = len(graph.x)
    stiffness = torch.zeros((node_count, node_count), dtype=torch.float32)
    source, target = graph.edge_index.long()
    stiffness[source, target] = graph.edge_attr[:, 3].float()
    stiffness[target, source] = graph.edge_attr[:, 3].float()
    return base, stiffness[base[0], base[1]]


def prepare_records(data: dict, ae, latent_model, arm: str, smoke: bool) -> tuple[list[dict], list[dict]]:
    records = {"train": [], "validation": []}
    limits = {"train": 4, "validation": 2} if smoke else {"train": 60, "validation": 20}
    for split in records:
        rows = [row for row in data["train" if split == "train" else "validation"] if row["source"] == "lj_noisy"]
        assert len(rows) == (60 if split == "train" else 20)
        for row in rows[:limits[split]]:
            positions = normalized_positions(row)
            torch.testing.assert_close(positions[0], row["ref_pos"], rtol=0, atol=2e-6)
            base, base_stiffness = base_relation(row)
            score = latent_score(ae, latent_model, row, data)
            latent_edges = select_topk_nonphysical_edges(score, base, 4)
            generator = torch.Generator().manual_seed(123_000 + int(row["original_index"]))
            random_edges = random_nonphysical_edges(
                len(positions[0]), base, latent_edges.shape[1], generator=generator
            )
            extra = {"base": base[:, :0], "random": random_edges, "latent": latent_edges}[arm]
            index = torch.cat((base, extra), dim=1)
            stiffness = torch.cat((base_stiffness, torch.zeros(extra.shape[1])))
            is_base = torch.cat((torch.ones(base.shape[1]), torch.zeros(extra.shape[1])))
            is_added = 1.0 - is_base
            records[split].append({
                "original_index": int(row["original_index"]),
                "positions": positions,
                "row": row,
                "edge_index": index,
                "stiffness": stiffness,
                "is_base": is_base,
                "is_added": is_added,
                "base_edge_count": int(base.shape[1]),
                "added_edge_count": int(extra.shape[1]),
                "latent_edge_count": int(latent_edges.shape[1]),
                "latent_score_mean": float(score[latent_edges[0], latent_edges[1]].mean()),
            })
            print(json.dumps({"stage": "edges", "arm": arm, "split": split,
                              "id": int(row["original_index"]), "base": base.shape[1],
                              "added": extra.shape[1]}), flush=True)
    return records["train"], records["validation"]


def node_features(positions: torch.Tensor, frame: int) -> torch.Tensor:
    return torch.cat((
        positions[frame] - positions[0],
        positions[frame] - positions[frame - 1],
        positions[frame - 1] - positions[frame - 2],
        positions[frame - 2] - positions[frame - 3],
    ), dim=-1)


def acceleration(positions: torch.Tensor, frame: int) -> torch.Tensor:
    return positions[frame + 1] - 2.0 * positions[frame] + positions[frame - 1]


def edge_features(record: dict, current: torch.Tensor) -> torch.Tensor:
    return geometric_edge_features(
        record["positions"][0], current, record["edge_index"], record["stiffness"],
        record["is_base"], record["is_added"],
    )


def moments(chunks: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    count = sum(chunk.shape[0] for chunk in chunks)
    total = sum((chunk.double().sum(0) for chunk in chunks), torch.zeros(chunks[0].shape[1], dtype=torch.double))
    square = sum((chunk.double().square().sum(0) for chunk in chunks), torch.zeros(chunks[0].shape[1], dtype=torch.double))
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-12)
    return mean.float(), variance.sqrt().float()


def fit_normalizers(train: list[dict]) -> dict[str, torch.Tensor]:
    node_chunks, target_chunks, edge_chunks = [], [], []
    for record in train:
        for frame in range(3, 100):
            node_chunks.append(node_features(record["positions"], frame))
            target_chunks.append(acceleration(record["positions"], frame))
        for frame in VALIDATION_FRAMES:
            edge_chunks.append(edge_features(record, record["positions"][frame]))
    node_mean, node_std = moments(node_chunks)
    edge_mean, edge_std = moments(edge_chunks)
    target_mean, target_std = moments(target_chunks)
    # Relation indicators must remain semantically distinct even if an arm has no added edges.
    edge_std[-2:] = 1.0
    return {"node_mean": node_mean, "node_std": node_std.clamp_min(1e-6),
            "edge_mean": edge_mean, "edge_std": edge_std.clamp_min(1e-6),
            "target_mean": target_mean, "target_std": target_std.clamp_min(1e-7)}


def batch_examples(examples: list[tuple[dict, int]], normalizers: dict) -> tuple[torch.Tensor, ...]:
    nodes, edges, indices, targets = [], [], [], []
    offset = 0
    for record, frame in examples:
        positions = record["positions"]
        node = (node_features(positions, frame) - normalizers["node_mean"]) / normalizers["node_std"]
        edge = (edge_features(record, positions[frame]) - normalizers["edge_mean"]) / normalizers["edge_std"]
        target = (acceleration(positions, frame) - normalizers["target_mean"]) / normalizers["target_std"]
        nodes.append(node); edges.append(edge); targets.append(target)
        indices.append(record["edge_index"] + offset)
        offset += len(node)
    return torch.cat(nodes), torch.cat(edges), torch.cat(indices, dim=1), torch.cat(targets)


def chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


@torch.no_grad()
def validate(model, records: list[dict], normalizers: dict, batch_size: int) -> float:
    model.eval()
    examples = [(record, frame) for record in records for frame in VALIDATION_FRAMES]
    values = []
    for batch in chunks(examples, batch_size):
        node, edge, index, target = batch_examples(batch, normalizers)
        values.append((model(node, edge, index) - target).square().mean().item())
    return float(np.mean(values))


def train(model, train_records: list[dict], validation_records: list[dict], normalizers: dict,
          smoke: bool, seed: int) -> tuple[dict, list[dict]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = random.Random(seed)
    epochs = 1 if smoke else 45
    frames_per = 1 if smoke else 2
    batch_size = 2 if smoke else 4
    history, best, best_state, bad = [], math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        examples = []
        for record in train_records:
            for frame in rng.sample(range(3, 100), frames_per):
                examples.append((record, frame))
        rng.shuffle(examples)
        losses = []
        for batch in chunks(examples, batch_size):
            node, edge, index, target = batch_examples(batch, normalizers)
            loss = (model(node, edge, index) - target).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        if smoke or epoch == 1 or epoch % 3 == 0:
            score = validate(model, validation_records, normalizers, batch_size)
            record = {"epoch": epoch, "train_acceleration_mse": float(np.mean(losses)),
                      "validation_acceleration_mse": score}
            history.append(record)
            print(json.dumps(record), flush=True)
            if score < best - 1e-5:
                best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if not smoke and bad >= 5:
                break
    assert best_state is not None
    return best_state, history


def batched_rollout(model, records: list[dict], normalizers: dict) -> list[torch.Tensor]:
    predictions = [record["positions"][:4].clone() for record in records]
    model.eval()
    with torch.no_grad():
        for frame in range(3, 100):
            examples, synthetic_records = [], []
            for record, path in zip(records, predictions):
                synthetic = dict(record)
                synthetic["positions"] = torch.cat((path, record["positions"][len(path):]), dim=0)
                synthetic_records.append(synthetic)
                examples.append((synthetic, frame))
            node, edge, index, _ = batch_examples(examples, normalizers)
            normalized_acceleration = model(node, edge, index)
            sizes = [len(record["positions"][0]) for record in records]
            split_acceleration = normalized_acceleration.split(sizes)
            for position_path, predicted_normalized in zip(predictions, split_acceleration):
                predicted_acceleration = predicted_normalized * normalizers["target_std"] + normalizers["target_mean"]
                next_position = 2.0 * position_path[-1] - position_path[-2] + predicted_acceleration
                position_path.resize_(len(position_path) + 1, position_path.shape[1], position_path.shape[2])
                position_path[-1] = next_position
    return predictions


def physical_positions(record: dict, normalized: torch.Tensor) -> torch.Tensor:
    row = record["row"]
    return normalized.double() * row["half"] + row["center"]


def evaluate_rollout(records: list[dict], predictions: list[torch.Tensor], arm: str, seed: int) -> pd.DataFrame:
    rows = []
    for record, prediction in zip(records, predictions):
        row = record["row"]
        reference = row["physical_reference"]
        sides = directional_side_indices_from_box(reference, quantile=0.10)
        for frame in HORIZONS:
            true_physical = row["physical_positions"][frame]
            pred_physical = physical_positions(record, prediction[frame])
            true_graph, pred_graph = clone_graph(reference), clone_graph(reference)
            true_graph.x = true_graph.x.clone(); pred_graph.x = pred_graph.x.clone()
            true_graph.x[:, :2] = true_physical; pred_graph.x[:, :2] = pred_physical
            true_p = float(calc_p_ratio_rollout_sides([reference, true_graph], -1, side_idx=sides))
            pred_p = float(calc_p_ratio_rollout_sides([reference, pred_graph], -1, side_idx=sides))
            rows.append({"arm": arm, "seed": seed, "split": "validation",
                         "original_index": record["original_index"], "frame": frame,
                         "true_p_ratio": true_p, "pred_p_ratio": pred_p,
                         "finite_pair": math.isfinite(true_p) and math.isfinite(pred_p),
                         "coordinate_mse": float((pred_physical - true_physical).square().mean()),
                         "base_edge_count": record["base_edge_count"],
                         "added_edge_count": record["added_edge_count"]})
    return pd.DataFrame(rows)


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.square(truth - truth.mean()).sum()
    return float(1.0 - np.square(truth - prediction).sum() / denominator) if denominator > 0 else float("nan")


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame, group in table.groupby("frame"):
        valid = group[group.finite_pair]
        rows.append({"arm": group.arm.iloc[0], "seed": int(group.seed.iloc[0]), "split": "validation",
                     "frame": int(frame), "p_ratio_r2": r2(valid.true_p_ratio.to_numpy(), valid.pred_p_ratio.to_numpy()),
                     "p_ratio_valid": len(valid), "p_ratio_total": len(group),
                     "coordinate_mse": float(group.coordinate_mse.mean()),
                     "added_edges_mean": float(group.added_edge_count.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("base", "random", "latent"), required=True)
    parser.add_argument("--seed", type=int, choices=(123, 456, 786), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    protocol = json.loads((CODE / "protocol.json").read_text())
    cache_path = ROOT / protocol["frozen_inputs"]["cache"]
    latent_path = ROOT / protocol["frozen_inputs"]["latent_propagator"]
    assert sha(cache_path) == protocol["frozen_inputs"]["cache_sha256"]
    assert sha(latent_path) == protocol["frozen_inputs"]["latent_propagator_sha256"]
    manifest = json.loads((CODE / "manifest.json").read_text())
    assert {name: sha(CODE / name) for name in manifest} == manifest
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert sha(Path(data["ae_checkpoint"])) == protocol["frozen_inputs"]["autoencoder_sha256"]
    ae = load_ae(data)
    latent_model = make_trajectory_propagator("history_attention", "mean")
    latent_bundle = torch.load(latent_path, map_location="cpu", weights_only=False)
    latent_model.load_state_dict(latent_bundle["state_dict"], strict=True)
    latent_model.eval()
    for parameter in latent_model.parameters():
        parameter.requires_grad_(False)
    frozen_before = {"ae": state_digest(ae.state_dict()), "latent": state_digest(latent_model.state_dict())}
    run_name = f"{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else "")
    output = OUT / "runs" / run_name
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    recipe = {"arm": args.arm, "seed": args.seed, "smoke": args.smoke, "job_id": os.environ.get("PBS_JOBID", "local"),
              "protocol": protocol, "protocol_sha256": sha(CODE / "protocol.json"),
              "code_manifest_sha256": sha(CODE / "manifest.json"), "final_test_used": False,
              "response_supervision_or_selection": False, "frozen_before": frozen_before}
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    try:
        train_records, validation_records = prepare_records(data, ae, latent_model, args.arm, args.smoke)
        normalizers = fit_normalizers(train_records)
        model = NormalizedAccelerationGNN(hidden_dim=64, steps=4)
        state, history = train(model, train_records, validation_records, normalizers, args.smoke, args.seed)
        model.load_state_dict(state)
        predictions = batched_rollout(model, validation_records, normalizers)
        table = evaluate_rollout(validation_records, predictions, args.arm, args.seed)
        summary = summarize(table)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        table.to_csv(output / "per_network_rows.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        torch.save({"state_dict": state, "normalizers": normalizers}, output / "model.pt")
        frozen_after = {"ae": state_digest(ae.state_dict()), "latent": state_digest(latent_model.state_dict())}
        assert frozen_before == frozen_after
        recipe.update({"frozen_after": frozen_after, "seconds": time.time() - started,
                       "checkpoint_sha256": sha(output / "model.pt"), "summary": summary.to_dict("records")})
        (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        (output / "completed.json").write_text(json.dumps({"status": "completed", "job_id": recipe["job_id"],
                                                            "seconds": time.time() - started}, indent=2) + "\n")
        print(summary.to_string(index=False), flush=True)
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}",
                                                         "job_id": recipe["job_id"]}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
