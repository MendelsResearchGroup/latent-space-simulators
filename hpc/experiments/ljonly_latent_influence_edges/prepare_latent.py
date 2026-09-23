#!/usr/bin/env python3
"""Cache LJ-only AE trajectories and fit a frozen 4D latent propagator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
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
OUT = ROOT / "notebooks/results/ljonly_latent_influence_edges"
LEGACY = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2"
sys.path.insert(0, str(LEGACY / "src"))

from graph_utils.box import Box
from lss.graph import clone_graph
from lss.latent.experiment import _load_ae_cache
from lss.latent.training import (
    compact_reference_edge_features,
    encode_frame_latent,
    reference_positions_for_model,
)

trajectory_spec = importlib.util.spec_from_file_location(
    "frozen_trajectory_propagator", CODE / "trajectory_propagator.py"
)
assert trajectory_spec is not None and trajectory_spec.loader is not None
trajectory_module = importlib.util.module_from_spec(trajectory_spec)
trajectory_spec.loader.exec_module(trajectory_module)
make_trajectory_propagator = trajectory_module.make_trajectory_propagator

SUPERVISION = (1, 2, 3, 5, 10, 25, 50, 75, 100)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(state: dict[str, torch.Tensor]) -> str:
    value = hashlib.sha256()
    for key in sorted(state):
        value.update(key.encode())
        value.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def decode(ae, record: dict, normalized_z: torch.Tensor, stats: dict) -> torch.Tensor:
    batch = torch.zeros(len(record["h_ref"]), dtype=torch.long)
    z = normalized_z.reshape(1, -1) * stats["z_std"] + stats["z_mean"]
    return ae.decode(z, record["h_ref"], batch)


def build_records(result: dict, smoke: bool) -> tuple[list[dict], list[dict], dict]:
    ae, norm, params = result["ae"], result["normalizers"], result["params"]
    groups = []
    for split, simulations, limit in (
        ("train", result["train_data"], 4 if smoke else 60),
        ("validation", result["val_data"], 2 if smoke else 20),
    ):
        records = []
        for simulation in simulations[:limit]:
            reference_graph = simulation[0]
            reference = reference_graph.x[:, :2].float()
            span = (reference.amax(0) - reference.amin(0)).clamp_min(1e-6)
            reference_edges = compact_reference_edge_features(reference_graph, pos_dim=2, device="cpu")
            normalized_edges = (reference_edges - norm["ref_edge_mean"]) / norm["ref_edge_std"]
            h_ref = ae.encode_reference_graph(
                reference_positions_for_model(reference_graph, pos_dim=2, device="cpu"),
                normalized_edges,
                reference_graph.edge_index.long(),
            ).detach()
            z = torch.stack([
                encode_frame_latent(ae, simulation, frame, pos_dim=2,
                                    node_feature_mode=params["node_feature_mode"],
                                    normalizers=norm, device="cpu")
                for frame in range(101)
            ]).detach()
            positions = torch.stack([frame.x[:, :2].float() for frame in simulation[:101]])
            target_norm = ((positions - reference) / span - norm["target_mean"]) / norm["target_std"]
            half = reference_graph.reference_box_half_extent.double()[:2]
            center = reference_graph.reference_box_center.double()[:2]
            physical_positions = positions.double() * half + center
            physical_reference = clone_graph(reference_graph).cpu()
            physical_reference.x = physical_reference.x.double().clone()
            physical_reference.x[:, :2] = physical_positions[0]
            box = reference_graph.box
            physical_reference.box = Box(
                float(box.x1 * half[0] + center[0]), float(box.x2 * half[0] + center[0]),
                float(box.y1 * half[1] + center[1]), float(box.y2 * half[1] + center[1]), -.1, .1,
            )
            physical_reference.box_tensor = reference_graph.box_tensor.double()[:2] * half
            physical_reference.coordinate_normalization = "physical"
            records.append({
                "source": "lj_noisy", "original_index": int(reference_graph.source_sim_id),
                "split": split, "latent_raw": z, "h_ref": h_ref,
                "reference_tokens": torch.cat((h_ref, reference), dim=-1).detach(),
                "target_norm": target_norm.detach(), "physical_positions": physical_positions,
                "physical_reference": physical_reference, "ref_pos": reference,
                "span": span, "half": half, "center": center,
            })
            print(json.dumps({"stage": "encode", "split": split,
                              "id": int(reference_graph.source_sim_id)}), flush=True)
        groups.append(records)
    train, validation = groups
    all_train_z = torch.cat([record["latent_raw"] for record in train])
    stats = {"z_mean": all_train_z.mean(0), "z_std": all_train_z.std(0, unbiased=False).clamp_min(1e-6)}
    for record in train + validation:
        record["latent_norm"] = (record.pop("latent_raw") - stats["z_mean"]) / stats["z_std"]
    return train, validation, stats


def losses(ae, model, record: dict, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    path = model(record["latent_norm"][0], record["reference_tokens"])
    latent = (path[list(SUPERVISION)] - record["latent_norm"][list(SUPERVISION)]).square().mean()
    decoded = torch.stack([decode(ae, record, path[frame], stats) for frame in SUPERVISION])
    position = (decoded - record["target_norm"][list(SUPERVISION)]).square().mean()
    return position, latent


@torch.no_grad()
def validate(ae, model, validation: list[dict], stats: dict) -> float:
    model.eval()
    return float(np.mean([float(losses(ae, model, record, stats)[0]) for record in validation]))


def fit_latent(ae, train: list[dict], validation: list[dict], stats: dict, smoke: bool):
    model = make_trajectory_propagator("history_attention", "mean", latent_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = random.Random(123)
    history, best, best_state, bad = [], float("inf"), None, 0
    for epoch in range(1, 2 if smoke else 61):
        model.train(); order = list(train); rng.shuffle(order); training = []
        for record in order:
            position, latent = losses(ae, model, record, stats)
            objective = position + 0.1 * latent
            optimizer.zero_grad(set_to_none=True); objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); training.append(float(objective.detach()))
        score = validate(ae, model, validation, stats)
        row = {"epoch": epoch, "training_objective": float(np.mean(training)),
               "validation_position_mse": score}
        history.append(row); print(json.dumps(row), flush=True)
        if score < best - 1e-5:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= (1 if smoke else 12):
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history, best


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(123); np.random.seed(123); torch.manual_seed(123); torch.set_num_threads(8)
    protocol = json.loads((CODE / "protocol.json").read_text())
    manifest = json.loads((CODE / "manifest.json").read_text())
    assert {name: sha(CODE / name) for name in manifest} == manifest
    checkpoint = ROOT / protocol["frozen_ae"]["path"]
    assert sha(checkpoint) == protocol["frozen_ae"]["sha256"]
    ae_dir = checkpoint.parent
    recipe = json.loads((ae_dir / "recipe.json").read_text())
    result = _load_ae_cache(checkpoint, recipe["config"], device="cpu")
    assert len(result["train_data"]) == 60 and len(result["val_data"]) == 20 and not result["test_data"]
    assert result["params"]["latent_dim"] == 4
    frozen_before = digest(result["ae"].state_dict())
    output = OUT / ("latent_smoke" if args.smoke else "latent")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        train, validation, stats = build_records(result, args.smoke)
        model, history, best = fit_latent(result["ae"], train, validation, stats, args.smoke)
        assert digest(result["ae"].state_dict()) == frozen_before
        cache = {"train": train, "validation": validation, "normalizers": result["normalizers"],
                 "latent_stats": stats, "ae_checkpoint": str(checkpoint),
                 "provenance": {"ae_sha256": sha(checkpoint), "ae_state_digest": frozen_before,
                                "train_count": len(train), "validation_count": len(validation),
                                "final_test_used": False}}
        torch.save(cache, output / "cache.pt")
        torch.save({"state_dict": model.state_dict(), "family": "history_attention",
                    "context_mode": "mean", "latent_dim": 4}, output / "latent_model.pt")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        completed = {"status": "completed", "job_id": os.environ.get("PBS_JOBID", "local"),
                     "seconds": time.time() - started, "best_validation_position_mse": best,
                     "cache_sha256": sha(output / "cache.pt"),
                     "latent_model_sha256": sha(output / "latent_model.pt"),
                     "ae_sha256": sha(checkpoint), "train_count": len(train),
                     "validation_count": len(validation), "final_test_used": False}
        (output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({"status": "failed",
            "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
