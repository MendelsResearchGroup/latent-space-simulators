#!/usr/bin/env python3
"""Nested training-count AE study with 20 complete heldout low-T paths per cell."""
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
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.training import encode_frame_latent

from evaluate import physical, pref, pratio


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/mini_temperature200_frames500"
MORPH = OUT / "latent_count_morph"
FROZEN_TRAIN = OUT / "code_v2/train.py"
ALL4 = ("node_low", "node_mixed", "stiffness_low", "stiffness_mixed")
LOW = ("node_low", "stiffness_low")
HELD = list(range(40, 120))
PATH_IDS = list(range(40, 60))
FRAMES = 500


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_module():
    spec = importlib.util.spec_from_file_location("mt200_frozen_train", FROZEN_TRAIN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def heldout(result: dict, manifest: dict, name: str) -> list:
    entry = manifest["datasets"][name]
    spec = {"name": name, "label": name, "path": entry["cache_path"], "train_count": 0, "val_count": 80, "split_indices": {"train": [], "val": HELD, "test": []}}
    source = {"name": name, "label": name, "path": entry["cache_path"], "dataset_name": name, "source_name": name, "coordinate_normalization": "position_normalization", "dataset_mixture": [spec]}
    _, data, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    assert len(data) == 80 and [int(sim[0].source_sim_id) for sim in data] == HELD
    return data


def metrics(z: np.ndarray) -> tuple[float, float]:
    centered = z - z.mean(0)
    covariance = centered.T @ centered
    eigenvalues = np.linalg.eigvalsh(covariance)
    effective = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    redundancy = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
    return effective, redundancy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    assert args.count in (1, 2, 4, 8, 10, 12, 14, 20)
    assert args.seed in (123, 456, 786)
    assert args.reuse == (args.seed == 123 and args.count in (1, 20))
    started = time.monotonic()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    seed_everything(args.seed)
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    folder = MORPH / f"c{args.count}_s{args.seed}"
    folder.mkdir(parents=True, exist_ok=False)
    frozen = train_module()
    if args.reuse:
        donor = OUT / ("all4_one_each" if args.count == 1 else "all4_twenty_each")
        recipe = json.loads((donor / "propagator_recipe.json").read_text())
        result = load_experiment_bundle(donor / "propagator.pt", cfg=recipe["config"], device="cpu")
        checkpoint = donor / "ae.pt"
    else:
        cfg = frozen.config(folder, args.seed)
        frozen.assert_ae_config(cfg, folder, args.seed)
        source = frozen.source_spec("all4_twenty_each", list(ALL4), manifest, args.count, False)
        result = run_latent_experiment(source, cfg, device="cpu")
        result["ae_history"].to_csv(folder / "ae_warmup_history.csv", index=False)
        frozen.fit_uniform(result, folder, 60)
        checkpoint = folder / "ae.pt"

    initial_rows, path_rows, summary_rows = [], [], []
    for name in LOW:
        simulations = heldout(result, manifest, name)
        source_initial = []
        for trajectory_id, simulation in zip(HELD, simulations):
            z = encode_frame_latent(result["ae"], simulation, 0, pos_dim=2, node_feature_mode=result["params"]["node_feature_mode"], normalizers=result["normalizers"], device="cpu").detach().cpu().reshape(-1)
            physical_reference, _ = pref(simulation[0])
            sides = directional_side_indices_from_box(physical_reference, quantile=0.10)
            final_graph = physical(simulation[FRAMES - 1], simulation[0])
            p_ratio = pratio(physical_reference, final_graph, sides)
            row = {"count": args.count, "seed": args.seed, "source": name, "trajectory_id": trajectory_id, "z0": float(z[0]), "z1": float(z[1]), "final_endpoint_p_ratio": p_ratio}
            source_initial.append(row)
            initial_rows.append(row)
            if trajectory_id in PATH_IDS:
                for frame in range(FRAMES):
                    latent = encode_frame_latent(result["ae"], simulation, frame, pos_dim=2, node_feature_mode=result["params"]["node_feature_mode"], normalizers=result["normalizers"], device="cpu").detach().cpu().reshape(-1)
                    path_rows.append({"count": args.count, "seed": args.seed, "source": name, "trajectory_id": trajectory_id, "frame": frame, "md_step": frame * 200, "z0": float(latent[0]), "z1": float(latent[1])})
        table = pd.DataFrame(source_initial)
        effective, redundancy = metrics(table[["z0", "z1"]].to_numpy())
        summary_rows.append({
            "count": args.count, "seed": args.seed, "source": name,
            "effective_dimension": effective, "z0_z1_pearson": redundancy,
            "initial_z0_final_p_abs_pearson": abs(table.z0.corr(table.final_endpoint_p_ratio)),
            "initial_z1_final_p_abs_pearson": abs(table.z1.corr(table.final_endpoint_p_ratio)),
            "valid": len(table), "total": 80,
        })
        del simulations
        gc.collect()
    pd.DataFrame(initial_rows).to_csv(folder / "initial_latents.csv", index=False)
    pd.DataFrame(path_rows).to_csv(folder / "trajectory_latents_20_ids.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(folder / "summary.csv", index=False)
    completed = {
        "status": "completed", "count": args.count, "seed": args.seed,
        "reused_existing_checkpoint": args.reuse, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "training_ids": list(range(args.count)), "validation_ids_by_source": frozen.ALL4_VALIDATION,
        "heldout_ids": HELD, "full_path_ids": PATH_IDS, "frames": [0, 499], "md_steps": [0, 99800],
        "sources_fitted": list(ALL4), "sources_plotted": list(LOW), "response_used_for_training_or_selection": False,
        "data_manifest_sha256": sha(OUT / "data_manifest.json"), "script_sha256": sha(Path(__file__)),
        "job_id": os.environ.get("PBS_JOBID"), "seconds": time.monotonic() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (folder / "completed.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completed, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
