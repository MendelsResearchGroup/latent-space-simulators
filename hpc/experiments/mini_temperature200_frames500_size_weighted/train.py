#!/usr/bin/env python3
"""Frozen, stiffness-low-only comparison of uniform and displacement-size AE loss."""
from __future__ import annotations

import argparse, copy, hashlib, json, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lss.dynamics.experiment import evaluate_autoencoder_reconstruction_horizons, resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.simulation import ae_target_tensor, batch_delta_graphs

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT_ROOT = ROOT / "notebooks/results/mini_temperature200_frames500_size_weighted/v2"
DATA_ROOT = ROOT / "notebooks/results/mini_temperature200_frames500"
CODE = Path(__file__).resolve().parent
FRAMES, SOURCE = 500, "stiffness_low"
ARMS = ("uniform", "inverse_energy_cap100")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_spec(manifest: dict, smoke: bool) -> dict:
    train = [0] if smoke else list(range(20))
    validation = [20] if smoke else list(range(20, 40))
    source = {"name": SOURCE, "label": SOURCE, "path": manifest["datasets"][SOURCE]["cache_path"],
              "train_count": len(train), "val_count": len(validation),
              "split_indices": {"train": train, "val": validation, "test": []}}
    # `resolve_dataset_splits` only honors explicit indices for mixture members.
    return {"name": SOURCE, "label": SOURCE, "path": source["path"], "dataset_name": SOURCE,
            "source_name": SOURCE, "train_count": len(train), "val_count": len(validation),
            "coordinate_normalization": "position_normalization", "dataset_mixture": [source]}


def manifest_path() -> Path:
    """Use frozen snapshot data after submission, original prepared manifest for preflight."""
    snapshot_manifest = CODE / "data_manifest.json"
    return snapshot_manifest if snapshot_manifest.exists() else DATA_ROOT / "data_manifest.json"


def preflight_resolved_splits(manifest: dict, smoke: bool) -> None:
    """Exercise the actual loader path and fail if explicit IDs were not honored."""
    source = source_spec(manifest, smoke)
    cfg = config(Path("/tmp/size-weighted-preflight"), 123)
    train, validation, heldout, info = resolve_train_val_test(source, {**cfg, **source}, split_seed=123)
    expected_train = [0] if smoke else list(range(20))
    expected_validation = [20] if smoke else list(range(20, 40))
    assert [int(sim[0].source_sim_id) for sim in train] == expected_train
    assert [int(sim[0].source_sim_id) for sim in validation] == expected_validation
    assert not heldout and len(info) == 1 and info[0]["excluded"] == 120-len(expected_train)-len(expected_validation)


def config(out: Path, seed: int) -> dict:
    # start order excludes exact-zero reference frame from normalizer fitting and warmup.
    return {"ae_config": {"model": "message_passing_reference16", "latent_dim": 2,
            "latent_tokens": 32, "hidden_size": 96, "edge_feature_dim": 4,
            "message_passing_steps": 2, "node_feature_mode": "normalized_delta",
            "target_mode": "normalized_delta", "max_train_frames_per_sim": FRAMES,
            "max_val_frames_per_sim": FRAMES, "max_epochs": 1, "patience": 1,
            "lr": 1e-4, "weight_decay": 1e-5, "mix_sources": False,
            "balance_sources": False, "gradient_method": "source_mean",
            "checkpoint_metric": "val_max_source_reconstruction", "checkpoint_mode": "min",
            "pratio_eval_every": 0}, "coordinate_normalization": "position_normalization",
            "edge_mode": "compact_stored", "static_context_use_physical_reference": True,
            "pos_dim": 2, "batch_graphs": 32, "frame_skip": 1,
            "train_frame_start_order": 1, "early_stop_min_delta": 1e-8,
            "split_seed": seed, "model_seed": seed, "should_train_propagator": False,
            "should_rollout": False, "force_train": True,
            "cache_require_matching_config": True, "cache_path": str(out / "ae.pt")}


def frame_energy_and_weights(train, params, norm, arm: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Training-only E_t in exactly the standardized target coordinates of loss."""
    def batch(rows):
        return batch_delta_graphs(train, rows, pos_dim=2, device="cpu",
            node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"])
    values = []
    with torch.no_grad():
        for frame in range(1, FRAMES):
            packed = batch([(index, frame) for index in range(len(train))])
            # This is a size diagnostic, not the centered reconstruction residual:
            # it measures reference displacement magnitude in target-scaled units.
            target = ae_target_tensor(packed, params["ae_target_mode"]) / norm["target_std"]
            values.append(target.square().mean())
    energy = torch.stack(values)
    epsilon = float(torch.median(energy).item() / 100.0)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive training displacement energy is required for inverse-energy weighting")
    if arm == "uniform":
        weights = torch.ones_like(energy)
    elif arm == "inverse_energy_cap100":
        weights = 1.0 / energy.clamp_min(epsilon)
        weights = weights / weights.mean()
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return energy, weights, epsilon


@torch.no_grad()
def frozen_validation_diagnostics(model, validation, params, norm, inverse_weights: torch.Tensor) -> tuple[pd.DataFrame, dict]:
    """Common post-fit coordinate diagnostics for both arms; never checkpoint inputs."""
    rows = []
    model.eval()
    for frame in range(1, FRAMES):
        packed = batch_delta_graphs(validation, [(index, frame) for index in range(len(validation))], pos_dim=2,
            device="cpu", node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"])
        target = (ae_target_tensor(packed, params["ae_target_mode"]) - norm["target_mean"]) / norm["target_std"]
        prediction, _ = model((packed["node_feature"]-norm["node_feature_mean"])/norm["node_feature_std"], packed["ref_pos"],
            (packed["edge_attr"]-norm["edge_mean"])/norm["edge_std"],
            (packed["ref_edge_attr"]-norm["ref_edge_mean"])/norm["ref_edge_std"], packed["edge_index"], packed["batch"])
        standardized_mse = (prediction-target).square().mean()
        predicted_position = packed["ref_pos"] + (prediction * norm["target_std"] + norm["target_mean"]) * packed["position_scale"]
        physical_mse = (predicted_position-packed["cur_pos"]).square().mean()
        rows.append({"frame": frame, "standardized_coordinate_mse": float(standardized_mse),
                     "physical_coordinate_mse": float(physical_mse), "inverse_energy_weight": float(inverse_weights[frame-1])})
    framewise = pd.DataFrame(rows)
    summary = {"common_unweighted_validation_mse": float(framewise["standardized_coordinate_mse"].mean()),
               "common_inverse_energy_validation_mse": float((framewise["standardized_coordinate_mse"] * framewise["inverse_energy_weight"]).mean()),
               "early_frames_1_to_50_unweighted_validation_mse": float(framewise.loc[framewise.frame <= 50, "standardized_coordinate_mse"].mean()),
               "early_frames_1_to_50_physical_coordinate_mse": float(framewise.loc[framewise.frame <= 50, "physical_coordinate_mse"].mean()),
               "diagnostic_only": True}
    return framewise, summary


def fit(result: dict, out: Path, arm: str, epochs: int) -> str:
    model, norm, params = result["ae"], result["normalizers"], result["params"]
    train, validation = result["train_data"], result["val_data"]
    for parameter in model.parameters(): parameter.requires_grad_(True)
    energy, weights, epsilon = frame_energy_and_weights(train, params, norm, arm)
    assert weights.numel() == FRAMES - 1 and torch.isclose(weights.mean(), torch.tensor(1.0))
    dump(out / "frame_weights.json", {"source": SOURCE, "energy_standardized_target": energy.tolist(),
        "weights_frames_1_to_499": weights.tolist(), "epsilon": epsilon, "cap": 100,
        "formula": "1/max(E_t, median(E_positive)/100), normalized to mean 1", "training_only": True})

    def objective(sims, rows):
        if any(frame == 0 for _, frame in rows): raise ValueError("frame 0 is excluded from this study")
        packed = batch_delta_graphs(sims, rows, pos_dim=2, device="cpu",
            node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"])
        target = (ae_target_tensor(packed, params["ae_target_mode"]) - norm["target_mean"]) / norm["target_std"]
        prediction, _ = model((packed["node_feature"]-norm["node_feature_mean"])/norm["node_feature_std"], packed["ref_pos"],
            (packed["edge_attr"]-norm["edge_mean"])/norm["edge_std"],
            (packed["ref_edge_attr"]-norm["ref_edge_mean"])/norm["ref_edge_std"], packed["edge_index"], packed["batch"])
        error = (prediction-target).square().mean(-1)
        return torch.stack([error[packed["batch"] == j].mean() * weights[frame-1]
                            for j, (_, frame) in enumerate(rows)]).mean()

    train_rows = [(index, frame) for index in range(len(train)) for frame in range(1, FRAMES)]
    def validate() -> float:
        model.eval(); rows = [(index, frame) for index in range(len(validation)) for frame in range(1, FRAMES)]
        with torch.no_grad():
            total = 0.0
            for start in range(0, len(rows), 32):
                part = rows[start:start+32]
                total += float(objective(validation, part)) * len(part)
            return total / len(rows)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best, state, bad = validate(), copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "validation_coordinate_objective": best}]
    for epoch in range(1, epochs + 1):
        random.shuffle(train_rows); model.train(); total = 0.0
        for start in range(0, len(train_rows), 32):
            rows = train_rows[start:start+32]; loss = objective(train, rows)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(rows)
        score = validate(); history.append({"epoch": epoch, "train_coordinate_objective": total/len(train_rows),
                                             "validation_coordinate_objective": score})
        pd.DataFrame(history).to_csv(out / "ae_history.csv", index=False)
        if score < best - 1e-8: best, state, bad = score, copy.deepcopy(model.state_dict()), 0
        else: bad += 1
        if bad >= 12: break
    model.load_state_dict(state)
    # Derive common diagnostics after restoring the frozen selected checkpoint.
    inverse_energy, inverse_weights, _ = frame_energy_and_weights(train, params, norm, "inverse_energy_cap100")
    framewise, diagnostics = frozen_validation_diagnostics(model, validation, params, norm, inverse_weights)
    assert len(framewise) == FRAMES - 1 and int(framewise.frame.min()) == 1
    framewise.to_csv(out / "frozen_validation_coordinate_framewise.csv", index=False)
    dump(out / "frozen_validation_common_diagnostics.json", diagnostics)
    pratio_rows, pratio_summary = evaluate_autoencoder_reconstruction_horizons(
        model, validation, cfg=params, normalizers=norm, dataset=SOURCE, split_name="validation_diagnostic",
        rollout_steps=list(range(1, FRAMES)), device="cpu")
    pratio_rows.to_csv(out / "frozen_validation_pratio_framewise.csv", index=False)
    pratio_summary.to_csv(out / "frozen_validation_pratio_summary.csv", index=False)
    bundle = torch.load(out / "ae.pt", map_location="cpu", weights_only=False)
    bundle.update({"ae_state_dict": state, "ae_history": history,
                   "size_weighted_recipe": {"arm": arm, "frame0_excluded": True,
                       "validation_objective_matches_training": True,
                       "postfit_pratio_diagnostic_not_selection": True}})
    torch.save(bundle, out / "ae.pt")
    return sha(out / "ae.pt")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=(123,456,786), default=123); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--assert-config", action="store_true"); parser.add_argument("--preflight-splits", action="store_true"); args = parser.parse_args()
    if args.assert_config:
        for seed in (123,456,786):
            cfg = config(Path("/tmp/size-weighted") / str(seed), seed)
            assert cfg["train_frame_start_order"] == 1 and cfg["ae_config"]["target_mode"] == "normalized_delta"
        print("size-weighted static assertions passed"); return
    seed_everything(args.seed)
    manifest = json.loads(manifest_path().read_text())
    assert manifest["status"] == "complete" and manifest["cached_prefix_frames"] == FRAMES
    assert sha(Path(manifest["datasets"][SOURCE]["cache_path"])) == manifest["datasets"][SOURCE]["cache_sha256"]
    if args.preflight_splits:
        preflight_resolved_splits(manifest, args.smoke)
        print("size-weighted actual split preflight passed", flush=True)
        return
    suffix = f"seed{args.seed}_smoke" if args.smoke else f"seed{args.seed}"
    out = OUT_ROOT / args.arm / suffix
    out.mkdir(parents=True, exist_ok=False); source = source_spec(manifest, args.smoke); cfg = config(out, args.seed)
    dump(out / "recipe.json", {"arm": args.arm, "seed": args.seed, "smoke": args.smoke, "config": cfg, "source": source,
        "frames_used": list(range(1, FRAMES)), "heldout_development_ids_untouched": list(range(40,120)),
        "protocol_sha256": sha(CODE / "protocol.json"), "data_manifest_sha256": sha(CODE / "data_manifest.json"),
        "response_used_for_training_or_selection": False})
    result = run_latent_experiment(source, cfg, device="cpu")
    result["ae_history"].to_csv(out / "ae_warmup_history.csv", index=False)
    checkpoint_sha = fit(result, out, args.arm, 1 if args.smoke else 60)
    dump(out / "completed.json", {"status": "completed", "ae_sha256": checkpoint_sha})


if __name__ == "__main__": main()
