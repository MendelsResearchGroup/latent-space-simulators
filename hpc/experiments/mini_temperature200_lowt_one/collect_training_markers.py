#!/usr/bin/env python3
"""Collect frozen ID-0 display markers for the selected low-T AE seed."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch

from graph_utils import directional_side_indices_from_box
from lss.dynamics import experiment
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import encode_frame_latent


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "notebooks/results/mini_temperature200_frames500"
RESULTS = ROOT / "notebooks/results/mini_temperature200_lowt_one"
LOW_SOURCES = ("node_low", "stiffness_low")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def one_simulation(params: dict, manifest: dict, source_name: str):
    entry = manifest["datasets"][source_name]
    component = {
        "name": source_name,
        "label": source_name,
        "path": entry["cache_path"],
        "train_count": 0,
        "val_count": 1,
        "split_indices": {"train": [], "val": [0], "test": []},
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
    assert len(simulations) == 1 and int(simulations[0][0].source_sim_id) == 0
    return simulations[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(123, 456, 786))
    args = parser.parse_args()

    folder = RESULTS / f"seed{args.seed}"
    checkpoint = folder / "ae.pt"
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loaded = experiment._load_ae_cache(checkpoint, bundle["params"], device="cpu")
    params = loaded["params"]
    model = loaded["ae"]
    normalizers = loaded["normalizers"]
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    evaluator = load_module("mt200_frozen_evaluate", BASE / "code_v6/evaluate.py")

    rows = []
    for source_name in LOW_SOURCES:
        simulation = one_simulation(params, manifest, source_name)
        latent = encode_frame_latent(
            model,
            simulation,
            0,
            pos_dim=2,
            node_feature_mode=params["node_feature_mode"],
            normalizers=normalizers,
            device="cpu",
        ).detach().cpu().reshape(-1)
        reference, _ = evaluator.pref(simulation[0])
        sides = directional_side_indices_from_box(reference, quantile=0.10)
        final_graph = evaluator.physical(simulation[499], simulation[0])
        rows.append({
            "seed": args.seed,
            "source": source_name,
            "trajectory_id": 0,
            "temperature_K": float(simulation[0].temperature),
            "initial_z0": float(latent[0]),
            "initial_z1": float(latent[1]),
            "training_p_ratio": evaluator.pratio(reference, final_graph, sides),
            "estimator": "physical endpoint, frame 499",
        })

    output = RESULTS / f"training_markers_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"{output}\nsha256={digest}")


if __name__ == "__main__":
    main()
