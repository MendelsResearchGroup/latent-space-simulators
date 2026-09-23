"""Freeze the all-four AE and fit a low-temperature-only propagator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

from lss.dynamics.experiment import run_latent_experiment, seed_everything


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/all4_ae_low_prop_1000"
DATA_OUT = ROOT / "notebooks/results/all4_coverage_1000"
SOURCE_MODEL = DATA_OUT / "all4_twenty_each_retry_v5_validation30"
SOURCE_AE = SOURCE_MODEL / "ae.pt"
CODE = Path(__file__).resolve().parent
ARM = "all4_ae_prop_low_retry1"
LOW_SOURCES = ("node_low", "stiffness_low")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_spec(manifest: dict, smoke: bool) -> dict:
    train_ids = [0] if smoke else list(range(20))
    val_ids = [20] if smoke else list(range(20, 40))
    mixture = []
    for name in LOW_SOURCES:
        mixture.append({
            "name": name,
            "label": name,
            "path": manifest["datasets"][name]["trainval_path"],
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "split_indices": {"train": train_ids, "val": val_ids, "test": []},
        })
    return {
        "name": "mini_shared",
        "dataset_name": "mini_shared",
        "source_name": "mini_shared",
        "label": "mini_shared",
        "path": mixture[0]["path"],
        "dataset_mixture": mixture,
        "coordinate_normalization": "position_normalization",
    }


def config(folder: Path, smoke: bool) -> dict:
    baseline = json.loads((SOURCE_MODEL / "propagator_recipe.json").read_text())["config"]
    cfg = copy.deepcopy(baseline)
    cfg.update({
        "cache_path": str(folder / "propagator.pt"),
        "pretrained_ae_cache_path": str(SOURCE_AE),
        "should_train_propagator": True,
        "should_rollout": False,
        "force_train": True,
        "dyn_max_epochs": 1 if smoke else 60,
        "dyn_patience": 1 if smoke else 12,
        "propagator_context_include_temperature": False,
        "propagator_context_include_source_id": False,
        "propagator_checkpoint_metric": "val_loss",
        "propagator_checkpoint_mode": "min",
        "propagator_rollout_eval_every_epoch": False,
        "rollout_eval_splits": [],
    })
    expected = copy.deepcopy(baseline)
    expected.update({
        "cache_path": str(folder / "propagator.pt"),
        "pretrained_ae_cache_path": str(SOURCE_AE),
        "dyn_max_epochs": 1 if smoke else 60,
        "dyn_patience": 1 if smoke else 12,
    })
    assert cfg == expected, "Hybrid changed a propagator setting beyond paths/smoke budget."
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--assert-config", action="store_true")
    args = parser.parse_args()
    assert SOURCE_AE.is_file()
    source_ae_sha = sha(SOURCE_AE)
    if args.assert_config:
        for smoke in (True, False):
            config(OUT / (ARM + ("_smoke" if smoke else "")), smoke)
        print("hybrid config parity passed", source_ae_sha, flush=True)
        return

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    seed_everything(123)
    folder = OUT / (ARM + ("_smoke" if args.smoke else ""))
    folder.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    source = source_spec(manifest, args.smoke)
    cfg = config(folder, args.smoke)
    for name in LOW_SOURCES:
        entry = manifest["datasets"][name]
        assert sha(Path(entry["trainval_path"])) == entry["trainval_sha256"]
    dump(folder / "recipe.json", {
        "status": "fitting",
        "arm": ARM,
        "job_id": os.environ.get("PBS_JOBID"),
        "seed": 123,
        "source": source,
        "config": cfg,
        "frozen_ae_path": str(SOURCE_AE),
        "frozen_ae_sha256_before": source_ae_sha,
        "ae_exposure": "node_low,node_mixed,stiffness_low,stiffness_mixed IDs0..19",
        "propagator_exposure": "node_low,stiffness_low IDs0..19 only",
        "selection": "low-T IDs20..39 latent-delta validation MSE",
        "heldout_development": "all four sources IDs40..119",
        "protocol_sha256": sha(CODE / "protocol.json"),
        "code_manifest_sha256": sha(CODE / "manifest.json"),
        "data_manifest_sha256": sha(OUT / "data_manifest.json"),
    })
    result = run_latent_experiment(source, cfg, device="cpu")
    assert sha(SOURCE_AE) == source_ae_sha
    frozen = torch.load(SOURCE_AE, map_location="cpu", weights_only=False)
    assert all(torch.equal(value.cpu(), frozen["ae_state_dict"][key].cpu()) for key, value in result["ae"].state_dict().items())
    result["dyn_history"].to_csv(folder / "propagator_history.csv", index=False)
    exposure = []
    for split, simulations in (("train", result["train_data"]), ("validation", result["val_data"])):
        expected_ids = ([0] if args.smoke else list(range(20))) if split == "train" else ([20] if args.smoke else list(range(20, 40)))
        for source_name in LOW_SOURCES:
            source_simulations = [simulation for simulation in simulations if str(simulation[0].source_name) == source_name]
            assert len(source_simulations) == len(expected_ids)
            for trajectory_id, simulation in zip(expected_ids, source_simulations):
                graph = simulation[0]
                exposure.append({
                    "split": split,
                    "source": source_name,
                    "trajectory_id": trajectory_id,
                    "temperature": float(graph.temperature),
                    "frames": len(simulation),
                    "first_time": int(simulation[0].time),
                    "last_time": int(simulation[-1].time),
                })
                assert all(frame_graph.time == frame * 1000 for frame, frame_graph in enumerate(simulation))
    pd.DataFrame(exposure).to_csv(folder / "exposure.csv", index=False)
    dump(folder / "fit_completed.json", {
        "status": "completed",
        "job_id": os.environ.get("PBS_JOBID"),
        "frozen_ae_sha256_before": source_ae_sha,
        "frozen_ae_sha256_after": sha(SOURCE_AE),
        "propagator_sha256": sha(folder / "propagator.pt"),
        "response_used_for_training_or_selection": False,
    })
    print("hybrid propagator fit complete", flush=True)


if __name__ == "__main__":
    main()
