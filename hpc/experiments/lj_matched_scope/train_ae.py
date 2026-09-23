#!/usr/bin/env python3
"""Matched four-trajectory AE scope comparison for noisy-LJ transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
BASE_CODE = ROOT / "notebooks/results/latent_matched_study/code_v1"
OUT = ROOT / "notebooks/results/lj_matched_scope"
sys.path.insert(0, str(BASE_CODE / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("ljonly4", "shared4"), required=True)
    parser.add_argument("--seed", type=int, choices=(123, 456, 786), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest_path = BASE_CODE / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    suffix = "_smoke" if args.smoke else ""
    output = OUT / f"{args.scope}_d2_s{args.seed}{suffix}"
    output.mkdir(parents=True, exist_ok=False)

    selected = ("reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy") if args.scope == "shared4" else ("lj_noisy",)
    mixture = []
    for name in selected:
        raw = manifest["sources"][name]
        if sha(Path(raw["path"])) != raw["sha256"]:
            raise ValueError(f"Dataset changed: {name}")
        if args.smoke:
            train_ids = raw["split_indices"]["train"][:1]
            val_ids = raw["split_indices"]["val"][:1] if name == "lj_noisy" else []
        elif args.scope == "ljonly4":
            train_ids = raw["split_indices"]["train"][:4]
            val_ids = raw["split_indices"]["val"][:20]
        else:
            train_ids = raw["split_indices"]["train"][:1]
            val_ids = raw["split_indices"]["val"][:20] if name == "lj_noisy" else []
        mixture.append({
            "name": name,
            "label": name,
            "path": raw["path"],
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "split_indices": {"train": train_ids, "val": val_ids, "test": []},
            "append_lj_indicator": True,
            "lj_max_graph_distance": 3 if name == "lj_noisy" else None,
        })

    frames = 4 if args.smoke else 101
    rows_per_source = 8 if args.smoke else (12120 if args.scope == "ljonly4" else 3030)
    source = {
        "dataset_name": "lj_matched_scope",
        "source_name": args.scope,
        "label": args.scope,
        "path": mixture[0]["path"],
        "dataset_mixture": mixture,
    }
    cfg = {
        "split_seed": 123,
        "model_seed": args.seed,
        "pos_dim": 2,
        "batch_graphs": 32,
        "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_mode": "compact_stored",
        "static_context_use_physical_reference": True,
        "should_rollout": False,
        "should_train_propagator": False,
        "force_train": True,
        "cache_path": str(output / "ae.pt"),
        "early_stop_min_delta": 1e-5,
        "ae_config": {
            "model": "attention",
            "latent_dim": 2,
            "latent_tokens": 32,
            "hidden_size": 128,
            "target_mode": "normalized_delta",
            "node_feature_mode": "normalized_delta",
            "edge_feature_dim": 5,
            "max_train_frames_per_sim": frames,
            "max_val_frames_per_sim": frames,
            "train_frame_skip": 1,
            "val_frame_skip": 1,
            "max_epochs": 1 if args.smoke else 40,
            "patience": 1 if args.smoke else 8,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "mix_sources": True,
            "balance_sources": True,
            "train_rows_per_source": rows_per_source,
            "gradient_method": "source_mean",
            "checkpoint_metric": "val_max_source_reconstruction",
            "checkpoint_mode": "min",
        },
    }
    recipe = {
        "scope": args.scope,
        "seed": args.seed,
        "smoke": args.smoke,
        "source": source,
        "config": cfg,
        "question": "Does cross-family exposure improve a matched four-trajectory 2D AE for unseen noisy-LJ graphs?",
        "matching": {
            "unique_training_trajectories": len(selected) if args.scope == "shared4" else 4,
            "frame_samples_per_epoch": rows_per_source * len(selected),
            "validation": "same 20 unseen noisy-LJ graphs; coordinate reconstruction only",
            "response_supervision": False,
            "final_test_used": False,
        },
        "manifest_sha256": sha(manifest_path),
        "runner_sha256": sha(Path(__file__)),
        "job_id": os.environ.get("PBS_JOBID", "local"),
        "host": platform.node(),
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    started = time.time()
    seed_everything(args.seed)
    result = run_latent_experiment(source, cfg, device="cpu")
    if result["test_data"]:
        raise AssertionError("Reserved final-test data were loaded")
    result["ae_history"].to_csv(output / "ae_history.csv", index=False)
    completed = {
        "status": "completed",
        "seconds": time.time() - started,
        "job_id": os.environ.get("PBS_JOBID", "local"),
        "ae_sha256": sha(output / "ae.pt"),
        "epochs": len(result["ae_history"]),
        "train_count": len(result["train_data"]),
        "validation_count": len(result["val_data"]),
        "final_test_used": False,
    }
    (output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")


if __name__ == "__main__":
    main()
