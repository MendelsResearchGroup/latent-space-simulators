#!/usr/bin/env python3
"""Train a 2D AE whose decoder target is the next stored-frame displacement."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_utils import directional_side_indices_from_box
from lss.dynamics.experiment import build_autoencoder, resolve_train_val_test, seed_everything
from lss.dynamics.simulation import (
    batch_delta_graphs,
    fit_edge_stats,
    fit_node_feature_stats,
    fit_reference_edge_stats,
)
from lss.dynamics.training import encode_frame_latent


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/mini_temperature200_frames500"
OUT = ROOT / "notebooks/results/mini_temperature200_one_step_ae"
FROZEN = BASE / "code_v6"
SOURCES = ("node_low", "stiffness_low")
FRAMES = 500
VALIDATION_IDS = list(range(20, 40))
HELDOUT_IDS = list(range(40, 120))
PATH_IDS = list(range(40, 60))
FULL_HORIZONS = (1, 5, 20, 100, 499)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def source_spec(manifest: dict, count: int, smoke: bool) -> dict:
    train_ids = [0] if smoke else list(range(count))
    val_ids = [20] if smoke else VALIDATION_IDS
    mixture = []
    for name in SOURCES:
        entry = manifest["datasets"][name]
        mixture.append({
            "name": name,
            "label": name,
            "path": entry["cache_path"],
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "split_indices": {"train": train_ids, "val": val_ids, "test": []},
        })
    return {
        "name": f"joint_low_step_count{count}",
        "label": f"joint_low_step_count{count}",
        "path": mixture[0]["path"],
        "dataset_name": "joint_low",
        "source_name": "joint_low",
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": mixture,
    }


def heldout(params: dict, manifest: dict, source_name: str, ids: list[int]):
    entry = manifest["datasets"][source_name]
    component = {
        "name": source_name,
        "label": source_name,
        "path": entry["cache_path"],
        "train_count": 0,
        "val_count": len(ids),
        "split_indices": {"train": [], "val": ids, "test": []},
    }
    source = {
        "name": source_name,
        "label": source_name,
        "path": entry["cache_path"],
        "dataset_name": source_name,
        "source_name": source_name,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [component],
    }
    _, simulations, _, _ = resolve_train_val_test(source, params, split_seed=123)
    assert [int(simulation[0].source_sim_id) for simulation in simulations] == ids
    return simulations


def step_target(simulations, rows, device="cpu") -> torch.Tensor:
    values = []
    for simulation_index, frame in rows:
        simulation = simulations[int(simulation_index)]
        reference = simulation[0].x[:, :2].to(device).float()
        current = simulation[int(frame)].x[:, :2].to(device).float()
        following = simulation[int(frame) + 1].x[:, :2].to(device).float()
        scale = (reference.max(0).values - reference.min(0).values).clamp_min(1e-6)
        values.append((following - current) / scale.reshape(1, -1))
    return torch.cat(values, dim=0)


def target_stats(simulations, rows, batch_graphs: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    total = torch.zeros(1, 2, dtype=torch.float64)
    square = torch.zeros(1, 2, dtype=torch.float64)
    for start in range(0, len(rows), batch_graphs):
        values = step_target(simulations, rows[start:start + batch_graphs]).double()
        count += len(values)
        total += values.sum(0, keepdim=True)
        square += values.square().sum(0, keepdim=True)
    mean = total / count
    variance = (square - total.square() / count) / max(count - 1, 1)
    return mean.float(), variance.clamp_min(0).sqrt().clamp_min(1e-6).float()


def packed(model, normalizers, simulations, rows):
    batch = batch_delta_graphs(
        simulations, rows, pos_dim=2, device="cpu",
        node_feature_mode="normalized_delta", edge_mode="compact_stored",
    )
    prediction, latent = model(
        (batch["node_feature"] - normalizers["node_feature_mean"]) / normalizers["node_feature_std"],
        batch["ref_pos"],
        (batch["edge_attr"] - normalizers["edge_mean"]) / normalizers["edge_std"],
        (batch["ref_edge_attr"] - normalizers["ref_edge_mean"]) / normalizers["ref_edge_std"],
        batch["edge_index"], batch["batch"],
    )
    target = (step_target(simulations, rows) - normalizers["target_mean"]) / normalizers["target_std"]
    return prediction, target, latent, batch


def source_loss(model, normalizers, simulations, rows, batch_graphs: int) -> float:
    total = 0.0
    with torch.no_grad():
        model.eval()
        for start in range(0, len(rows), batch_graphs):
            part = rows[start:start + batch_graphs]
            prediction, target, _, _ = packed(model, normalizers, simulations, part)
            total += float((prediction - target).square().mean()) * len(part)
    return total / len(rows)


def r2(true, pred) -> float:
    true = np.asarray(true, dtype=float); pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(true) & np.isfinite(pred)
    if valid.sum() < 2:
        return float("nan")
    true = true[valid]; pred = pred[valid]
    denominator = np.square(true - true.mean()).sum()
    return float(1 - np.square(true - pred).sum() / denominator) if denominator > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True, choices=(1, 20))
    parser.add_argument("--seed", type=int, required=True, choices=(123, 456, 786))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    seed_everything(args.seed)

    tag = "smoke" if args.smoke else f"count{args.count}_seed{args.seed}"
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    spec = source_spec(manifest, args.count, args.smoke)
    template = torch.load(BASE / "joint_low/ae.pt", map_location="cpu", weights_only=False)
    params = copy.deepcopy(template["params"])
    params.update({
        "model_seed": args.seed,
        "ae_target_mode": "normalized_next_step_delta",
        "name": spec["name"], "label": spec["label"], "path": spec["path"],
        "dataset_name": spec["dataset_name"], "source_name": spec["source_name"],
        "dataset_mixture": spec["dataset_mixture"],
    })
    train, validation, _, split_info = resolve_train_val_test(spec, params, split_seed=123)
    train_frames = 8 if args.smoke else FRAMES - 1
    train_rows = [(index, frame) for index in range(len(train)) for frame in range(train_frames)]
    validation_rows = {
        source: [(index, frame) for index, simulation in enumerate(validation)
                 if str(simulation[0].source_name) == source for frame in range(train_frames)]
        for source in SOURCES
    }
    batch_graphs = 32
    node_mean, node_std = fit_node_feature_stats(
        train, train_rows, pos_dim=2, batch_graphs=batch_graphs, device="cpu",
        node_feature_mode="normalized_delta",
    )
    edge_mean, edge_std = fit_edge_stats(
        train, train_rows, pos_dim=2, batch_graphs=batch_graphs, device="cpu", edge_mode="compact_stored",
    )
    ref_mean, ref_std = fit_reference_edge_stats(
        train, train_rows, pos_dim=2, batch_graphs=batch_graphs, device="cpu", edge_mode="compact_stored",
    )
    target_mean, target_std = target_stats(train, train_rows, batch_graphs)
    normalizers = {
        "node_feature_mean": node_mean, "node_feature_std": node_std,
        "edge_mean": edge_mean, "edge_std": edge_std,
        "ref_edge_mean": ref_mean, "ref_edge_std": ref_std,
        "target_mean": target_mean, "target_std": target_std,
    }
    params["node_feature_dim"] = int(node_mean.numel())
    model = build_autoencoder(params, edge_dim=int(edge_mean.numel()), device="cpu")
    model.edge_mode = "compact_stored"
    model.set_edge_normalization(edge_mean, edge_std, ref_mean, ref_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def validate():
        losses = {source: source_loss(model, normalizers, validation, rows, batch_graphs)
                  for source, rows in validation_rows.items()}
        return max(losses.values()), losses

    best, initial_losses = validate()
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "val_max_source_mse": best, **{f"val_{k}_mse": v for k, v in initial_losses.items()}}]
    patience = 1 if args.smoke else 12
    max_epochs = 1 if args.smoke else 60
    stale = 0
    for epoch in range(1, max_epochs + 1):
        random.shuffle(train_rows)
        model.train(); total = 0.0
        for start in range(0, len(train_rows), batch_graphs):
            rows = train_rows[start:start + batch_graphs]
            prediction, target, _, _ = packed(model, normalizers, train, rows)
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(rows)
        score, losses = validate()
        history.append({"epoch": epoch, "train_mse": total / len(train_rows),
                        "val_max_source_mse": score, **{f"val_{k}_mse": v for k, v in losses.items()}})
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(json.dumps(history[-1]), flush=True)
        if score < best - 1e-8:
            best = score; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state); model.eval()
    bundle = {
        "kind": "one_step_displacement_ae", "params": params, "source_spec": spec,
        "normalizers": normalizers, "ae_state_dict": best_state, "history": history,
        "target_definition": "(x[t+1]-x[t]) / reference_box_extent",
    }
    torch.save(bundle, folder / "ae.pt")
    pd.DataFrame(split_info).to_csv(folder / "split_info.csv", index=False)

    ids = [40] if args.smoke else HELDOUT_IDS
    path_ids = [40] if args.smoke else (PATH_IDS if args.seed == 456 else [])
    horizons = (1, 5) if args.smoke else FULL_HORIZONS
    evaluator = load_module("mt200_step_evaluate", FROZEN / "evaluate.py")
    latent_rows, path_rows, prediction_rows = [], [], []
    for source in SOURCES:
        simulations = heldout(params, manifest, source, ids)
        for trajectory_id, simulation in zip(ids, simulations):
            reference, _ = evaluator.pref(simulation[0])
            sides = directional_side_indices_from_box(reference, quantile=0.10)
            final_graph = evaluator.physical(simulation[499], simulation[0])
            final_p = evaluator.pratio(reference, final_graph, sides)
            initial_z = encode_frame_latent(model, simulation, 0, pos_dim=2, node_feature_mode="normalized_delta", normalizers=normalizers, device="cpu").detach().reshape(-1)
            latent_rows.append({"seed": args.seed, "count": args.count, "source": source,
                                "trajectory_id": trajectory_id, "z0": float(initial_z[0]),
                                "z1": float(initial_z[1]), "final_endpoint_p_ratio": final_p})
            if trajectory_id in path_ids:
                for frame in range(FRAMES):
                    z = encode_frame_latent(model, simulation, frame, pos_dim=2, node_feature_mode="normalized_delta", normalizers=normalizers, device="cpu").detach().reshape(-1)
                    path_rows.append({"seed": args.seed, "count": args.count, "source": source,
                                      "trajectory_id": trajectory_id, "frame": frame,
                                      "md_step": frame * 200, "z0": float(z[0]), "z1": float(z[1])})
            for target_frame in horizons:
                current_frame = target_frame - 1
                prediction, target, _, batch = packed(model, normalizers, [simulation], [(0, current_frame)])
                predicted_delta = prediction * target_std + target_mean
                predicted_next = batch["cur_pos"] + predicted_delta * batch["position_scale"]
                predicted_graph = simulation[target_frame].clone()
                predicted_graph.x = predicted_graph.x.clone()
                predicted_graph.x[:, :2] = predicted_next.detach()
                predicted_physical = evaluator.physical(predicted_graph, simulation[0])
                truth_physical = evaluator.physical(simulation[target_frame], simulation[0])
                prediction_rows.append({
                    "seed": args.seed, "count": args.count, "source": source,
                    "trajectory_id": trajectory_id, "target_frame": target_frame,
                    "md_step": target_frame * 200,
                    "normalized_step_mse": float((prediction - target).square().mean()),
                    "true_p_ratio": evaluator.pratio(reference, truth_physical, sides),
                    "pred_p_ratio": evaluator.pratio(reference, predicted_physical, sides),
                })
        del simulations; gc.collect()

    initial = pd.DataFrame(latent_rows); predictions = pd.DataFrame(prediction_rows)
    initial.to_csv(folder / "initial_latents.csv", index=False)
    pd.DataFrame(path_rows).to_csv(folder / "trajectory_latents_20_ids.csv", index=False)
    predictions.to_csv(folder / "one_step_predictions.csv", index=False)
    summary_rows = []
    if not args.smoke:
        for (source, target_frame), table in predictions.groupby(["source", "target_frame"]):
            summary_rows.append({
                "seed": args.seed, "count": args.count, "source": source,
                "target_frame": target_frame, "md_step": target_frame * 200,
                "p_ratio_r2": r2(table.true_p_ratio, table.pred_p_ratio),
                "normalized_step_mse": table.normalized_step_mse.mean(),
                "valid": int((np.isfinite(table.true_p_ratio) & np.isfinite(table.pred_p_ratio)).sum()),
                "total": len(table),
            })
    pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False)
    completed = {
        "status": "completed", "count": args.count, "seed": args.seed, "smoke": args.smoke,
        "checkpoint_sha256": sha(folder / "ae.pt"), "training_sources": list(SOURCES),
        "training_ids_per_source": [0] if args.smoke else list(range(args.count)),
        "validation_ids_per_source": [20] if args.smoke else VALIDATION_IDS,
        "heldout_ids_per_source": ids, "path_ids": path_ids,
        "p_ratio_used_for_training_or_selection": False,
        "target_definition": "normalized x[t+1]-x[t]", "encoder_input": "normalized x[t]-x[0]",
        "seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    dump(folder / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
