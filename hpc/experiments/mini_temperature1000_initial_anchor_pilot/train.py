"""One-seed propagator-only test of explicit frame-zero latent conditioning."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from anchor_context import install

install()

from lss.dynamics.experiment import run_latent_experiment, seed_everything

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/mini_temperature1000_mixed_only/joint_mixed_only"
OUT = ROOT / "notebooks/results/mini_temperature1000_initial_anchor_pilot"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def recipe(smoke: bool) -> tuple[dict, dict]:
    source = copy.deepcopy(json.loads((BASE / "recipe.json").read_text())["source"])
    cfg = copy.deepcopy(json.loads((BASE / "propagator_recipe.json").read_text())["config"])
    destination = OUT / ("anchored_s123_smoke" if smoke else "anchored_s123")
    cfg.update(
        cache_path=str(destination / "propagator.pt"),
        pretrained_ae_cache_path=str(BASE / "ae.pt"),
        propagator_model="delta_initial_latent_mlp",
        graph_context_dim=16,
        propagator_context_include_initial_latent=True,
        force_train=True,
        should_train_propagator=True,
        should_rollout=False,
        rollout_eval_splits=[],
    )
    if smoke:
        for item in source["dataset_mixture"]:
            item["train_count"] = 1
            item["val_count"] = 1
            item["split_indices"] = {"train": [0], "val": [20], "test": []}
        cfg.update(dyn_max_epochs=1, dyn_patience=1,
                   dyn_max_train_transitions_per_sim=3,
                   dyn_max_val_transitions_per_sim=3)
    return source, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--assert-config", action="store_true")
    args = parser.parse_args()
    source, cfg = recipe(args.smoke)
    assert tuple(item["name"] for item in source["dataset_mixture"]) == (
        "stiffness_mixed", "node_mixed"
    )
    assert cfg["propagator_model"] == "delta_initial_latent_mlp"
    assert cfg["propagator_loss"] == "delta"
    assert cfg["propagator_objective"] == "one_step"
    assert cfg["propagator_use_static_context"] is True
    assert cfg["propagator_context_include_temperature"] is False
    assert cfg["propagator_context_include_source_id"] is False
    assert cfg["graph_context_dim"] == 16
    if args.assert_config:
        print("initial-anchor pilot configuration assertions passed", flush=True)
        return
    destination = Path(cfg["cache_path"]).parent
    destination.mkdir(parents=True, exist_ok=False)
    baseline_ae_sha = sha(BASE / "ae.pt")
    dump(destination / "recipe.json", {
        "status": "frozen",
        "seed": 123,
        "smoke": args.smoke,
        "source": source,
        "config": cfg,
        "baseline_ae_sha256": baseline_ae_sha,
        "intervention": "append normalized frozen frame-zero latent after the unchanged static-context projection at every step",
        "unchanged": "AE, split, current-z input, static context, delta loss, optimizer, and selection",
        "test_used_for_selection": False,
        "job_id": os.environ.get("PBS_JOBID", "local"),
    })
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    seed_everything(123)
    result = run_latent_experiment(source, cfg, device="cpu")
    assert sha(BASE / "ae.pt") == baseline_ae_sha
    result["dyn_history"].to_csv(destination / "propagator_history.csv", index=False)
    dump(destination / "fit_completed.json", {
        "status": "completed",
        "ae_unchanged": True,
        "baseline_ae_sha256": baseline_ae_sha,
        "propagator_sha256": sha(destination / "propagator.pt"),
    })


if __name__ == "__main__":
    main()
