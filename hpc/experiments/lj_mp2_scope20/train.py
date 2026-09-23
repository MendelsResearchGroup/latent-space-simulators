#!/usr/bin/env python3
"""Matched MP2D noisy-LJ regularization study: LJ-only60 versus shared3x20."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
CODE = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2"
OUT = ROOT / "notebooks/results/lj_mp2_scope20"
sys.path.insert(0, str(CODE / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("ljonly60", "shared3x20"), required=True)
    parser.add_argument("--seed", type=int, choices=(123, 456, 786), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    random.seed(args.seed); np.random.seed(args.seed); seed_everything(args.seed)

    template_path = ROOT / "notebooks/results/compact_lj_mp2d/mp2_r16_d2_s123_code_v4/recipe.json"
    template = json.loads(template_path.read_text())
    available = {spec["name"]: spec for spec in template["source"]["dataset_mixture"]}
    names = ("lj_noisy",) if args.scope == "ljonly60" else ("reid", "depablo_low_temp", "lj_noisy")
    train_counts = {"lj_noisy": 60} if args.scope == "ljonly60" else {
        "reid": 20, "depablo_low_temp": 20, "lj_noisy": 20,
    }
    mixture = []
    for name in names:
        spec = copy.deepcopy(available[name])
        count = 2 if args.smoke and args.scope == "ljonly60" else (1 if args.smoke else train_counts[name])
        val_count = 2 if args.smoke and name == "lj_noisy" else (20 if not args.smoke and name == "lj_noisy" else 0)
        spec["train_count"] = count
        spec["val_count"] = val_count
        spec["split_indices"]["train"] = spec["split_indices"]["train"][:count]
        spec["split_indices"]["val"] = spec["split_indices"]["val"][:val_count]
        spec["split_indices"]["test"] = []
        mixture.append(spec)

    source = {
        "dataset_name": "lj_mp2_scope20",
        "source_name": args.scope,
        "label": args.scope,
        "path": mixture[0]["path"],
        "dataset_mixture": mixture,
        "coordinate_normalization": "position_normalization",
    }
    cfg = copy.deepcopy(template["config"])
    output = OUT / f"{args.scope}_s{args.seed}" / ("smoke" if args.smoke else "full")
    output.mkdir(parents=True, exist_ok=False)
    cfg["model_seed"] = args.seed
    cfg["cache_path"] = str(output / "ae.pt")
    cfg["force_train"] = True
    cfg["should_rollout"] = False
    cfg["should_train_propagator"] = False
    cfg["ae_config"]["model"] = "message_passing_reference16"
    cfg["ae_config"]["latent_dim"] = 2
    cfg["ae_config"]["message_passing_steps"] = 2
    cfg["ae_config"]["max_train_frames_per_sim"] = 4 if args.smoke else 200
    cfg["ae_config"]["max_val_frames_per_sim"] = 4 if args.smoke else 200
    cfg["ae_config"]["max_epochs"] = 1 if args.smoke else 60
    cfg["ae_config"]["patience"] = 1 if args.smoke else 12
    cfg["ae_config"]["balance_sources"] = False
    cfg["ae_config"]["checkpoint_metric"] = "val_max_source_reconstruction"

    recipe = {
        "scope": args.scope, "seed": args.seed, "smoke": args.smoke,
        "source": source, "config": cfg,
        "matching": {
            "training_trajectories": sum(item["train_count"] for item in mixture),
            "training_frames_per_epoch": sum(item["train_count"] for item in mixture) * (4 if args.smoke else 200),
            "validation": "same 20 unseen noisy-LJ trajectories only",
            "architecture": "message_passing_reference16, latent2, two message-passing steps",
            "response_supervision": False, "final_test_used": False,
        },
        "template_recipe": str(template_path), "template_sha256": sha(template_path),
        "runner_sha256": sha(Path(__file__)), "job_id": os.environ.get("PBS_JOBID", "local"),
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    started = time.time()
    result = run_latent_experiment(source, cfg, device="cpu")
    if result["test_data"] or len(result["train_data"]) != sum(item["train_count"] for item in mixture):
        raise RuntimeError("Split contract changed")
    if len(result["val_data"]) != (2 if args.smoke else 20):
        raise RuntimeError("Validation contract changed")
    result["ae_history"].to_csv(output / "ae_history.csv", index=False)
    completed = {
        "status": "completed", "seconds": time.time() - started,
        "job_id": os.environ.get("PBS_JOBID", "local"), "ae_sha256": sha(output / "ae.pt"),
        "epochs": len(result["ae_history"]), "train_count": len(result["train_data"]),
        "validation_count": len(result["val_data"]), "final_test_used": False,
    }
    (output / "completed.json").write_text(json.dumps(completed, indent=2) + "\n")


if __name__ == "__main__":
    main()
