#!/usr/bin/env python3
"""Low-T-only one-network-per-family AE and heldout latent export."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_utils import directional_side_indices_from_box
from lss.dynamics.experiment import resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.training import encode_frame_latent


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/mini_temperature200_frames500"
OUT = ROOT / "notebooks/results/mini_temperature200_lowt_one"
FROZEN = BASE / "code_v6"
FROZEN_TRAIN = FROZEN / "train.py"
FROZEN_EVALUATE = FROZEN / "evaluate.py"
PROTOCOL = Path(__file__).resolve().parent / "protocol.json"
LOW = ("node_low", "stiffness_low")
TRAIN_IDS = [0]
VALIDATION_IDS = list(range(20, 40))
HELDOUT_IDS = list(range(40, 120))
PATH_IDS = list(range(40, 60))
FRAMES = 500


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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


def heldout(result: dict, manifest: dict, name: str, ids: list[int]) -> list:
    entry = manifest["datasets"][name]
    component = {
        "name": name, "label": name, "path": entry["cache_path"],
        "train_count": 0, "val_count": len(ids),
        "split_indices": {"train": [], "val": ids, "test": []},
    }
    source = {
        "name": name, "label": name, "path": entry["cache_path"],
        "dataset_name": name, "source_name": name,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [component],
    }
    _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    assert [int(simulation[0].source_sim_id) for simulation in simulations] == ids
    return simulations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(123, 456, 786))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    seed_everything(args.seed)
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["cached_prefix_frames"] == FRAMES
    for name in LOW:
        entry = manifest["datasets"][name]
        assert sha(Path(entry["cache_path"])) == entry["cache_sha256"]

    folder = OUT / (("smoke_seed123") if args.smoke else f"seed{args.seed}")
    folder.mkdir(parents=True, exist_ok=False)
    frozen = load_module("mt200_frozen_train", FROZEN_TRAIN)
    evaluator = load_module("mt200_frozen_evaluate", FROZEN_EVALUATE)
    source = frozen.source_spec("joint_low", list(LOW), manifest, 1, False)
    assert [component["name"] for component in source["dataset_mixture"]] == list(LOW)
    assert all(component["split_indices"]["train"] == TRAIN_IDS for component in source["dataset_mixture"])
    assert all(component["split_indices"]["val"] == VALIDATION_IDS for component in source["dataset_mixture"])
    cfg = frozen.config(folder, args.seed)
    frozen.assert_ae_config(cfg, folder, args.seed)
    dump(folder / "recipe.json", {
        "seed": args.seed, "smoke": args.smoke, "source": source, "config": cfg,
        "training_ids": TRAIN_IDS, "validation_ids": VALIDATION_IDS,
        "heldout_development_ids": HELDOUT_IDS, "training_frames": [0, 499],
        "mixed_temperature_exposure": False, "propagator_trained": False,
        "response_used_for_training_or_selection": False,
        "protocol_sha256": sha(PROTOCOL), "data_manifest_sha256": sha(BASE / "data_manifest.json"),
        "frozen_base_manifest_sha256": sha(FROZEN / "manifest.json"),
        "job_id": os.environ.get("PBS_JOBID"),
    })

    result = run_latent_experiment(source, cfg, device="cpu")
    exposure = []
    for split, simulations, expected_ids in (
        ("train", result["train_data"], TRAIN_IDS),
        ("validation", result["val_data"], VALIDATION_IDS),
    ):
        for name in LOW:
            subset = [simulation for simulation in simulations if str(simulation[0].source_name) == name]
            assert [int(simulation[0].source_sim_id) for simulation in subset] == expected_ids
            for simulation in subset:
                exposure.append({
                    "split": split, "source": name,
                    "trajectory_id": int(simulation[0].source_sim_id),
                    "frames": FRAMES, "first_md_step": 0, "last_md_step": 99800,
                })
    pd.DataFrame(exposure).to_csv(folder / "exposure.csv", index=False)
    result["ae_history"].to_csv(folder / "ae_warmup_history.csv", index=False)
    checkpoint_sha = frozen.fit_uniform(result, folder, 1 if args.smoke else 60)

    ids = [40] if args.smoke else HELDOUT_IDS
    path_ids = [40] if args.smoke else PATH_IDS
    initial_rows, path_rows, summary_rows = [], [], []
    for name in LOW:
        simulations = heldout(result, manifest, name, ids)
        for trajectory_id, simulation in zip(ids, simulations):
            latent = encode_frame_latent(
                result["ae"], simulation, 0, pos_dim=2,
                node_feature_mode=result["params"]["node_feature_mode"],
                normalizers=result["normalizers"], device="cpu",
            ).detach().cpu().reshape(-1)
            reference, _ = evaluator.pref(simulation[0])
            sides = directional_side_indices_from_box(reference, quantile=0.10)
            final_graph = evaluator.physical(simulation[FRAMES - 1], simulation[0])
            final_p = evaluator.pratio(reference, final_graph, sides)
            initial_rows.append({
                "seed": args.seed, "source": name, "trajectory_id": trajectory_id,
                "z0": float(latent[0]), "z1": float(latent[1]),
                "final_endpoint_p_ratio": final_p,
            })
            if trajectory_id in path_ids:
                frames = [0, 1] if args.smoke else range(FRAMES)
                for frame in frames:
                    frame_latent = encode_frame_latent(
                        result["ae"], simulation, frame, pos_dim=2,
                        node_feature_mode=result["params"]["node_feature_mode"],
                        normalizers=result["normalizers"], device="cpu",
                    ).detach().cpu().reshape(-1)
                    path_rows.append({
                        "seed": args.seed, "source": name, "trajectory_id": trajectory_id,
                        "frame": frame, "md_step": 200 * frame,
                        "z0": float(frame_latent[0]), "z1": float(frame_latent[1]),
                    })
        del simulations
        gc.collect()

    initial = pd.DataFrame(initial_rows)
    if not args.smoke:
        for name, table in initial.groupby("source"):
            z0_p = table.z0.corr(table.final_endpoint_p_ratio)
            z1_p = table.z1.corr(table.final_endpoint_p_ratio)
            z0_s = table.z0.corr(table.final_endpoint_p_ratio, method="spearman")
            z1_s = table.z1.corr(table.final_endpoint_p_ratio, method="spearman")
            summary_rows.append({
                "seed": args.seed, "source": name,
                "z0_final_p_pearson": z0_p, "z1_final_p_pearson": z1_p,
                "strongest_abs_pearson": max(abs(z0_p), abs(z1_p)),
                "z0_final_p_spearman": z0_s, "z1_final_p_spearman": z1_s,
                "strongest_abs_spearman": max(abs(z0_s), abs(z1_s)),
                "valid": len(table), "total": 80,
            })
    initial.to_csv(folder / "initial_latents.csv", index=False)
    pd.DataFrame(path_rows).to_csv(folder / "trajectory_latents_20_ids.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False)
    completed = {
        "status": "completed", "seed": args.seed, "smoke": args.smoke,
        "checkpoint_sha256": checkpoint_sha,
        "training_sources": list(LOW), "training_ids_per_source": TRAIN_IDS,
        "validation_ids_per_source": VALIDATION_IDS,
        "heldout_ids_per_source": ids, "path_ids": path_ids,
        "mixed_temperature_exposure": False, "propagator_trained": False,
        "response_used_for_training_or_selection": False,
        "valid_per_source": 1 if args.smoke else 80,
        "protocol_sha256": sha(PROTOCOL), "data_manifest_sha256": sha(BASE / "data_manifest.json"),
        "script_sha256": sha(Path(__file__)), "job_id": os.environ.get("PBS_JOBID"),
        "seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    dump(folder / "completed.json", completed)
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
