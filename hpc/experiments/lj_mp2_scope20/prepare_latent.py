#!/usr/bin/env python3
"""Fit frozen-AE latent propagators for the matched MP2D scope-20 arms."""
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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
CODE = Path(os.environ["LSS_CODE_ROOT"]).resolve()
OUT = ROOT / "notebooks/results/lj_mp2_scope20"
LEGACY = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2"
sys.path.insert(0, str(LEGACY / "src"))

from lss.latent.experiment import _load_ae_cache
from lss.latent.training import (
    compact_reference_edge_features,
    encode_frame_latent,
    reference_positions_for_model,
)

trajectory_spec = importlib.util.spec_from_file_location(
    "frozen_trajectory_propagator", ROOT / "hpc/experiments/lj_matched_scope/trajectory_propagator.py"
)
assert trajectory_spec is not None and trajectory_spec.loader is not None
trajectory_module = importlib.util.module_from_spec(trajectory_spec)
trajectory_spec.loader.exec_module(trajectory_module)
make_trajectory_propagator = trajectory_module.make_trajectory_propagator

SUPERVISION = (1, 2, 3, 5, 10, 25, 50, 75, 100)
SCOPES = ("ljonly60", "shared3x20")
SEEDS = (123, 456, 786)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
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


def build_records(result: dict, recipe: dict, smoke: bool) -> tuple[list[dict], list[dict], dict]:
    ae, normalizers, params = result["ae"], result["normalizers"], result["params"]
    split_indices = {
        split: {
            item["name"]: list(item["split_indices"]["train" if split == "train" else "val"])
            for item in recipe["source"]["dataset_mixture"]
        }
        for split in ("train", "validation")
    }
    groups: list[list[dict]] = []
    for split, simulations, limit in (
        ("train", result["train_data"], 4 if smoke else 60),
        ("validation", result["val_data"], 2 if smoke else 20),
    ):
        records = []
        source_offsets: Counter = Counter()
        for simulation in simulations[:limit]:
            reference_graph = simulation[0]
            reference = reference_graph.x[:, :2].float()
            span = (reference.amax(0) - reference.amin(0)).clamp_min(1e-6)
            reference_edges = compact_reference_edge_features(reference_graph, pos_dim=2, device="cpu")
            normalized_edges = (reference_edges - normalizers["ref_edge_mean"]) / normalizers["ref_edge_std"]
            h_ref = ae.encode_reference_graph(
                reference_positions_for_model(reference_graph, pos_dim=2, device="cpu"),
                normalized_edges, reference_graph.edge_index.long(),
            ).detach()
            latent_raw = torch.stack([
                encode_frame_latent(ae, simulation, frame, pos_dim=2,
                                    node_feature_mode=params["node_feature_mode"],
                                    normalizers=normalizers, device="cpu")
                for frame in range(101)
            ]).detach()
            positions = torch.stack([frame.x[:, :2].float() for frame in simulation[:101]])
            target_norm = ((positions - reference) / span - normalizers["target_mean"]) / normalizers["target_std"]
            source = str(getattr(reference_graph, "source_name", "unknown"))
            source_offset = source_offsets[source]
            source_offsets[source] += 1
            original_index = int(split_indices[split][source][source_offset])
            records.append({
                "source": source, "original_index": original_index, "split": split,
                "latent_raw": latent_raw, "h_ref": h_ref,
                "reference_tokens": torch.cat((h_ref, reference), dim=-1).detach(),
                "target_norm": target_norm.detach(),
            })
            print(json.dumps({"stage": "encode", "split": split, "source": source,
                              "id": original_index}), flush=True)
        groups.append(records)
    train, validation = groups
    train_latent = torch.cat([record["latent_raw"] for record in train])
    stats = {"z_mean": train_latent.mean(0), "z_std": train_latent.std(0, unbiased=False).clamp_min(1e-6)}
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


def fit_latent(ae, train: list[dict], validation: list[dict], stats: dict, smoke: bool, seed: int):
    token_dim = int(train[0]["reference_tokens"].shape[1])
    if any(int(record["reference_tokens"].shape[1]) != token_dim for record in train + validation):
        raise ValueError("Reference-token width varies across frozen AE trajectories")
    model = make_trajectory_propagator("history_attention", "mean", latent_dim=2, token_dim=token_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = random.Random(seed)
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
    return model, history, best, token_dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(8)
    protocol = json.loads((CODE / "protocol.json").read_text())
    manifest = json.loads((CODE / "manifest.json").read_text())
    assert {name: sha(CODE / name) for name in manifest} == manifest
    checkpoint = OUT / f"{args.scope}_s{args.seed}" / "full" / "ae.pt"
    completed_ae = json.loads((checkpoint.parent / "completed.json").read_text())
    expected_ae_sha = protocol["frozen_autoencoders"][args.scope][str(args.seed)]["sha256"]
    assert completed_ae["status"] == "completed" and sha(checkpoint) == expected_ae_sha
    recipe = json.loads((checkpoint.parent / "recipe.json").read_text())
    result = _load_ae_cache(checkpoint, recipe["config"], device="cpu")
    expected_train = {item["name"]: item["train_count"] for item in recipe["source"]["dataset_mixture"]}
    expected_val = {item["name"]: item["val_count"] for item in recipe["source"]["dataset_mixture"] if item["val_count"]}
    actual_train = Counter(str(getattr(simulation[0], "source_name", "unknown")) for simulation in result["train_data"])
    actual_val = Counter(str(getattr(simulation[0], "source_name", "unknown")) for simulation in result["val_data"])
    assert not result["test_data"] and actual_train == expected_train and actual_val == expected_val
    assert result["params"]["latent_dim"] == 2
    frozen_before = digest(result["ae"].state_dict())
    output = OUT / "latent" / f"{args.scope}_s{args.seed}" / ("smoke" if args.smoke else "full")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        train, validation, stats = build_records(result, recipe, args.smoke)
        model, history, best, token_dim = fit_latent(result["ae"], train, validation, stats, args.smoke, args.seed)
        assert digest(result["ae"].state_dict()) == frozen_before
        torch.save({"train": train, "validation": validation, "normalizers": result["normalizers"],
                    "latent_stats": stats, "ae_checkpoint": str(checkpoint),
                    "provenance": {"ae_sha256": expected_ae_sha, "ae_state_digest": frozen_before,
                                   "train_sources": dict(actual_train), "validation_sources": dict(actual_val),
                                   "final_test_used": False}}, output / "cache.pt")
        torch.save({"state_dict": model.state_dict(), "family": "history_attention",
                    "context_mode": "mean", "latent_dim": 2, "token_dim": token_dim}, output / "latent_model.pt")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        completed = {"status": "completed", "job_id": os.environ.get("PBS_JOBID", "local"),
                     "seconds": time.time() - started, "best_validation_position_mse": best,
                     "cache_sha256": sha(output / "cache.pt"), "latent_model_sha256": sha(output / "latent_model.pt"),
                     "scope": args.scope, "seed": args.seed, "token_dim": token_dim,
                     "ae_sha256": expected_ae_sha, "train_sources": dict(actual_train),
                     "validation_sources": dict(actual_val), "final_test_used": False}
        (output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({"status": "failed", "scope": args.scope,
            "seed": args.seed, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
