#!/usr/bin/env python3
"""Frozen 500-frame AE and autonomous latent-propagator training matrix."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lss.dynamics.experiment import run_latent_experiment, seed_everything
from lss.dynamics.simulation import ae_target_tensor, batch_delta_graphs


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / os.environ.get("LSS_RESULT_DIR", "notebooks/results/mini_temperature200_frames500")
CODE = Path(__file__).resolve().parent
BASELINE = ROOT / "notebooks/results/all4_coverage_1000/all4_twenty_each_retry_v5_validation30"
TRAIN_FRAMES = int(os.environ.get("LSS_TRAIN_FRAMES", "500"))
CACHE_FRAMES = 500
ALL4 = ("node_low", "node_mixed", "stiffness_low", "stiffness_mixed")
ALL4_VALIDATION = {
    "node_low": list(range(20, 28)), "node_mixed": list(range(20, 28)),
    "stiffness_low": list(range(20, 27)), "stiffness_mixed": list(range(20, 27)),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def arm_spec(arm: str, protocol: dict) -> tuple[list[str], int]:
    spec = protocol["base_arms"][arm]
    return list(spec["sources"]), int(spec["train_per_source"])


def validation_ids(arm: str, source: str, smoke: bool) -> list[int]:
    if smoke:
        return [20]
    return ALL4_VALIDATION[source] if arm.startswith("all4_") else list(range(20, 40))


def source_spec(arm: str, names: list[str], manifest: dict, train_count: int, smoke: bool) -> dict:
    count = 1 if smoke else train_count
    mixture = []
    for name in names:
        ids = validation_ids(arm, name, smoke)
        mixture.append({
            "name": name, "label": name, "path": manifest["datasets"][name]["cache_path"],
            "train_count": count, "val_count": len(ids),
            "split_indices": {"train": list(range(count)), "val": ids, "test": []},
        })
    return {
        "name": arm, "label": arm, "path": mixture[0]["path"],
        "dataset_name": arm, "source_name": arm,
        "dataset_mixture": mixture, "coordinate_normalization": "position_normalization",
    }


def config(out: Path, seed: int) -> dict:
    cfg = json.loads((BASELINE / "recipe.json").read_text())["config"]
    cfg["ae_config"]["max_train_frames_per_sim"] = TRAIN_FRAMES
    cfg["ae_config"]["max_val_frames_per_sim"] = TRAIN_FRAMES
    cfg["cache_path"] = str(out / "ae.pt")
    cfg["model_seed"] = seed
    cfg["should_train_propagator"] = False
    cfg["should_rollout"] = False
    cfg["force_train"] = True
    return cfg


def assert_ae_config(cfg: dict, out: Path, seed: int) -> None:
    expected = json.loads((BASELINE / "recipe.json").read_text())["config"]
    expected["ae_config"]["max_train_frames_per_sim"] = TRAIN_FRAMES
    expected["ae_config"]["max_val_frames_per_sim"] = TRAIN_FRAMES
    expected["cache_path"] = str(out / "ae.pt")
    expected["model_seed"] = seed
    assert cfg == expected, "AE recipe differs beyond the declared frame horizon, seed and output path."


def propagator_config(cfg: dict, out: Path, smoke: bool) -> dict:
    dyn = copy.deepcopy(cfg)
    dyn.update(
        should_train_propagator=True,
        pretrained_ae_cache_path=str(out / "ae.pt"),
        pretrained_ae_config_keys=["autoencoder_model", "latent_dim", "latent_tokens", "hidden_size", "edge_feature_dim", "message_passing_steps", "node_feature_mode", "ae_target_mode", "edge_mode", "pos_dim"],
        pretrained_ae_require_matching_config=True,
        pretrained_ae_skip_stat_fitting=True,
        pretrained_ae_require_matching_normalizers=False,
        cache_path=str(out / "propagator.pt"),
        propagator_model="delta_mlp",
        propagator_loss="delta",
        propagator_objective="one_step",
        propagator_hidden_size=64,
        propagator_use_static_context=True,
        propagator_context_pool="mean",
        graph_context_dim=16,
        propagator_context_include_temperature=False,
        propagator_context_include_source_id=False,
        propagator_source_loss_reduction="equal",
        propagator_position_loss_weight=0.0,
        physics_loss_enabled=False,
        propagator_checkpoint_metric="val_loss",
        propagator_checkpoint_mode="min",
        propagator_rollout_eval_every_epoch=False,
        dyn_max_epochs=1 if smoke else 60,
        dyn_patience=1 if smoke else 12,
        dyn_lr=1e-4,
        dyn_weight_decay=1e-5,
        dyn_max_train_transitions_per_sim=TRAIN_FRAMES - 1,
        dyn_max_val_transitions_per_sim=TRAIN_FRAMES - 1,
        rollout_steps_grid=[TRAIN_FRAMES - 1],
        rollout_eval_splits=[],
    )
    expected = json.loads((BASELINE / "propagator_recipe.json").read_text())["config"]
    expected["ae_config"]["max_train_frames_per_sim"] = TRAIN_FRAMES
    expected["ae_config"]["max_val_frames_per_sim"] = TRAIN_FRAMES
    expected["cache_path"] = str(out / "propagator.pt")
    expected["pretrained_ae_cache_path"] = str(out / "ae.pt")
    expected["model_seed"] = cfg["model_seed"]
    expected["dyn_max_train_transitions_per_sim"] = TRAIN_FRAMES - 1
    expected["dyn_max_val_transitions_per_sim"] = TRAIN_FRAMES - 1
    expected["rollout_steps_grid"] = [TRAIN_FRAMES - 1]
    if smoke:
        expected["dyn_max_epochs"] = 1
        expected["dyn_patience"] = 1
    differing = {key: {"actual": dyn.get(key), "expected": expected.get(key)} for key in sorted(set(dyn) | set(expected)) if dyn.get(key) != expected.get(key)}
    assert not differing, f"Propagator recipe differs beyond declared horizon, seed, output paths and smoke budget: {differing}"
    return dyn


def fit_uniform(result: dict, out: Path, epochs: int) -> str:
    model, norm, params = result["ae"], result["normalizers"], result["params"]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    train, validation = result["train_data"], result["val_data"]
    names = sorted({str(sim[0].source_name) for sim in train})

    def batch(sims, rows):
        return batch_delta_graphs(sims, rows, pos_dim=2, device="cpu", node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"])

    energy = {}
    with torch.no_grad():
        for name in names:
            indices = [index for index, sim in enumerate(train) if str(sim[0].source_name) == name]
            values = []
            for frame in range(TRAIN_FRAMES):
                packed = batch(train, [(index, frame) for index in indices])
                target = ae_target_tensor(packed, params["ae_target_mode"]) / norm["target_std"]
                values.append(torch.stack([target[packed["batch"] == j].square().mean() for j in range(len(indices))]).mean())
            energy[name] = torch.stack(values)
    weights = {name: torch.ones_like(value) for name, value in energy.items()}
    assert all(torch.equal(value, torch.ones_like(value)) for value in weights.values())
    dump(out / "frame_weights.json", {
        name: {"energy_diagnostic": energy[name].tolist(), "weights": weights[name].tolist(), "post_warmup_weighting": "uniform_exactly_1_per_frame"}
        for name in names
    })

    def objective(sims, rows):
        packed = batch(sims, rows)
        target = (ae_target_tensor(packed, params["ae_target_mode"]) - norm["target_mean"]) / norm["target_std"]
        prediction, _ = model(
            (packed["node_feature"] - norm["node_feature_mean"]) / norm["node_feature_std"], packed["ref_pos"],
            (packed["edge_attr"] - norm["edge_mean"]) / norm["edge_std"],
            (packed["ref_edge_attr"] - norm["ref_edge_mean"]) / norm["ref_edge_std"],
            packed["edge_index"], packed["batch"],
        )
        error = (prediction - target).square().mean(-1)
        return torch.stack([
            error[packed["batch"] == j].mean() * weights[str(sims[index][0].source_name)][frame]
            for j, (index, frame) in enumerate(rows)
        ]).mean()

    train_rows = [(index, frame) for index in range(len(train)) for frame in range(TRAIN_FRAMES)]

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        scores = []
        for name in names:
            rows = [(index, frame) for index, sim in enumerate(validation) if str(sim[0].source_name) == name for frame in range(TRAIN_FRAMES)]
            total = 0.0
            for start in range(0, len(rows), 32):
                part = rows[start:start + 32]
                total += float(objective(validation, part)) * len(part)
            scores.append(total / len(rows))
        return float(np.mean(scores))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best, state, bad = validate(), copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "val_loss": best}]
    for epoch in range(1, epochs + 1):
        random.shuffle(train_rows)
        model.train()
        total = 0.0
        for start in range(0, len(train_rows), 32):
            rows = train_rows[start:start + 32]
            loss = objective(train, rows)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        score = validate()
        history.append({"epoch": epoch, "train_loss": total / len(train_rows), "val_loss": score})
        pd.DataFrame(history).to_csv(out / "ae_history.csv", index=False)
        print("uniform AE", epoch, score, flush=True)
        if score < best - 1e-8:
            best, state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= 12:
            break
    model.load_state_dict(state)
    bundle = torch.load(out / "ae.pt", map_location="cpu", weights_only=False)
    bundle["ae_state_dict"] = state
    bundle["ae_history"] = history
    bundle["frame_balance_recipe"] = {"training_only": True, "post_warmup_weighting": "uniform_exactly_1_per_frame", "warmup_epochs": 1}
    torch.save(bundle, out / "ae.pt")
    return sha(out / "ae.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(json.loads((CODE / "protocol.json").read_text())["base_arms"]))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--assert-config", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    protocol = json.loads((CODE / "protocol.json").read_text())
    if args.assert_config:
        for smoke in (False, True):
            out = OUT / ("static_smoke" if smoke else "static_production")
            cfg = config(out, args.seed)
            assert_ae_config(cfg, out, args.seed)
            propagator_config(cfg, out, smoke)
        print("500-frame AE and propagator parity assertions passed", flush=True)
        return
    seed_everything(args.seed)
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["cached_prefix_frames"] == CACHE_FRAMES
    assert 2 <= TRAIN_FRAMES <= CACHE_FRAMES
    names, train_count = arm_spec(args.arm, protocol)
    suffix = args.tag + ("_smoke" if args.smoke else "")
    out = OUT / f"{args.arm}{suffix}"
    out.mkdir(exist_ok=False)
    for name in names:
        assert sha(Path(manifest["datasets"][name]["cache_path"])) == manifest["datasets"][name]["cache_sha256"]
    source = source_spec(args.arm, names, manifest, train_count, args.smoke)
    cfg = config(out, args.seed)
    assert_ae_config(cfg, out, args.seed)
    dump(out / "recipe.json", {
        "config": cfg, "source": source, "arm": args.arm, "seed": args.seed,
        "job_id": os.environ.get("PBS_JOBID"), "protocol_sha256": sha(CODE / "protocol.json"),
        "data_manifest_sha256": sha(OUT / "data_manifest.json"),
        "training_ids": list(range(1 if args.smoke else train_count)),
        "validation_ids_by_source": {name: validation_ids(args.arm, name, args.smoke) for name in names},
        "heldout_development_ids": list(range(40, 120)), "training_frames": list(range(TRAIN_FRAMES)),
        "cached_frames": [0, CACHE_FRAMES - 1],
        "response_used_for_training_or_selection": False,
    })
    result = run_latent_experiment(source, cfg, device="cpu")
    expected_train = 1 if args.smoke else train_count
    exposure = []
    for split, simulations in (("train", result["train_data"]), ("validation", result["val_data"])):
        for name in names:
            subset = [sim for sim in simulations if str(sim[0].source_name) == name]
            expected = expected_train if split == "train" else len(validation_ids(args.arm, name, args.smoke))
            assert len(subset) == expected, (split, name, len(subset), expected)
            assert all(len(sim) == CACHE_FRAMES for sim in subset)
            assert all(int(sim[frame].time) == frame * 200 for sim in subset for frame in range(TRAIN_FRAMES))
            for simulation in subset:
                graph = simulation[0]
                exposure.append({
                    "split": split, "source": name, "trajectory_id": int(graph.source_sim_id),
                    "temperature_K": float(graph.temperature), "frames": TRAIN_FRAMES,
                    "first_md_step": int(simulation[0].time), "last_md_step": int(simulation[TRAIN_FRAMES - 1].time),
                })
    pd.DataFrame(exposure).to_csv(out / "exposure.csv", index=False)
    result["ae_history"].to_csv(out / "ae_warmup_history.csv", index=False)
    ae_sha = fit_uniform(result, out, 1 if args.smoke else 60)
    del result
    gc.collect()
    dump(out / "ae_completed.json", {"status": "completed", "sha256": ae_sha})
    dyn_cfg = propagator_config(cfg, out, args.smoke)
    dump(out / "propagator_recipe.json", {"config": dyn_cfg, "ae_sha256": ae_sha})
    seed_everything(args.seed)
    result = run_latent_experiment(source, dyn_cfg, device="cpu")
    frozen = torch.load(out / "ae.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(value.cpu(), frozen["ae_state_dict"][key].cpu()) for key, value in result["ae"].state_dict().items())
    assert sha(out / "ae.pt") == ae_sha
    result["dyn_history"].to_csv(out / "propagator_history.csv", index=False)
    dump(out / "fit_completed.json", {"status": "completed", "ae_unchanged": True, "propagator_sha256": sha(out / "propagator.pt")})


if __name__ == "__main__":
    main()
