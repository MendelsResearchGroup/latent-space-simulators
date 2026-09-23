"""Two-frame-conditioned stochastic latent rollout across unseen temperatures."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import base_train
from evaluate import physical, _pref
from graph_utils import directional_side_indices_from_box
from lss.dynamics.experiment import _load_ae_cache, resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.training import decode_latent_to_graph, encode_frame_latent, encode_reference_context


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
CODE = Path(os.environ["LSS_CODE_ROOT"])
OUT = ROOT / "notebooks/results/mixed_temperature_extrapolation_2frame"
DATA = ROOT / "notebooks/results/all4_coverage_1000/data_manifest.json"
BASE_RECIPE = ROOT / "notebooks/results/all4_coverage_1000/all4_twenty_each_retry_v5_validation30/recipe.json"
SOURCES = ("node_mixed", "stiffness_mixed")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def protocol():
    return json.loads((CODE / "protocol.json").read_text())


def source_spec(ids_train, ids_val):
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    mixture = []
    for name in SOURCES:
        path = manifest["datasets"][name]["trainval_path"]
        mixture.append({"name": name, "label": name, "path": path,
                        "train_count": len(ids_train), "val_count": len(ids_val),
                        "split_indices": {"train": ids_train, "val": ids_val, "test": []}})
    return {"name": "mixed_low_temperature", "dataset_name": "mixed_low_temperature",
            "source_name": "mixed_low_temperature", "label": "mixed_low_temperature",
            "path": mixture[0]["path"], "dataset_mixture": mixture,
            "coordinate_normalization": "position_normalization"}


def ae_config(folder):
    cfg = copy.deepcopy(json.loads(BASE_RECIPE.read_text())["config"])
    cfg.update(cache_path=str(folder / "ae.pt"), should_train_propagator=False,
               should_rollout=False, force_train=True)
    return cfg


def check_temperatures(simulations, allowed):
    observed = {float(simulation[0].temperature) for simulation in simulations}
    assert observed <= set(allowed), (observed, allowed)
    return sorted(observed)


class StochasticTwoFramePropagator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(21, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def distribution(self, current_z, previous_delta, context, progress):
        raw = self.network(torch.cat((current_z, previous_delta, context, progress[:, None]), dim=-1))
        return raw[:, :2], raw[:, 2:].clamp(-5.0, 1.5)


def load_ae(folder):
    recipe = json.loads((folder / "recipe.json").read_text())
    return _load_ae_cache(folder / "ae.pt", cfg=recipe["config"], device="cpu")


@torch.no_grad()
def encode_records(result, simulations, ids):
    records = []
    for trajectory_id, simulation in zip(ids, simulations):
        z = torch.stack([encode_frame_latent(result["ae"], simulation, frame, pos_dim=2,
                         node_feature_mode=result["params"]["node_feature_mode"],
                         normalizers=result["normalizers"], device="cpu") for frame in range(200)])
        context = encode_reference_context(result["ae"], simulation, pos_dim=2,
                    normalizers=result["normalizers"], device="cpu", include_temperature=False,
                    include_source_id=False, pool_mode="mean").reshape(-1)
        records.append({"source": str(simulation[0].source_name), "trajectory_id": int(trajectory_id),
                        "temperature_K": float(simulation[0].temperature), "z": z, "context": context,
                        "simulation": simulation})
    return records


def normalization(records):
    z = torch.cat([row["z"] for row in records])
    dz = torch.cat([row["z"][1:] - row["z"][:-1] for row in records])
    context = torch.stack([row["context"] for row in records])
    return {"z_mean": z.mean(0), "z_std": z.std(0).clamp_min(1e-6),
            "dz_mean": dz.mean(0), "dz_std": dz.std(0).clamp_min(1e-6),
            "context_mean": context.mean(0), "context_std": context.std(0).clamp_min(1e-6)}


def transition_tensors(records, stats):
    features, targets, sources = [], [], []
    for row in records:
        z = row["z"]
        context = ((row["context"] - stats["context_mean"]) / stats["context_std"]).expand(198, -1)
        current = (z[1:-1] - stats["z_mean"]) / stats["z_std"]
        previous = ((z[1:-1] - z[:-2]) - stats["dz_mean"]) / stats["dz_std"]
        target = ((z[2:] - z[1:-1]) - stats["dz_mean"]) / stats["dz_std"]
        progress = torch.arange(1, 199, dtype=z.dtype) / 199
        features.append(torch.cat((current, previous, context, progress[:, None]), -1))
        targets.append(target)
        sources.extend([row["source"]] * 198)
    return torch.cat(features), torch.cat(targets), np.asarray(sources)


def nll(model, features, target):
    mean, log_sd = model.distribution(features[:, :2], features[:, 2:4], features[:, 4:20], features[:, 20])
    return (0.5 * torch.square((target - mean) / log_sd.exp()) + log_sd).mean()


@torch.no_grad()
def validation_loss(model, features, targets, sources):
    model.eval(); values = []
    for source in SOURCES:
        index = np.flatnonzero(sources == source)
        chunks = [float(nll(model, features[part], targets[part])) for part in np.array_split(index, max(1, math.ceil(len(index) / 1024)))]
        values.append(float(np.mean(chunks)))
    return float(np.mean(values))


def fit_dynamics(train, val, stats, folder, smoke):
    train_x, train_y, train_s = transition_tensors(train, stats)
    val_x, val_y, val_s = transition_tensors(val, stats)
    model = StochasticTwoFramePropagator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best = validation_loss(model, val_x, val_y, val_s)
    state, best_epoch, bad = copy.deepcopy(model.state_dict()), 0, 0
    history = [{"epoch": 0, "train_nll": np.nan, "validation_nll": best}]
    generator = torch.Generator().manual_seed(123)
    epochs = 1 if smoke else 60
    patience = 1 if smoke else 12
    for epoch in range(1, epochs + 1):
        model.train(); order = torch.randperm(len(train_x), generator=generator); losses = []
        for index in order.split(512):
            loss = nll(model, train_x[index], train_y[index])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        score = validation_loss(model, val_x, val_y, val_s)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)), "validation_nll": score})
        pd.DataFrame(history).to_csv(folder / "dynamics_history.csv", index=False)
        print("dynamics", epoch, score, flush=True)
        if score < best - 1e-5:
            best, best_epoch, bad, state = score, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= patience: break
    model.load_state_dict(state)
    torch.save({"state_dict": state, "stats": stats, "best_epoch": best_epoch,
                "best_validation_nll": best, "observed_frames": [0, 1], "samples": 8}, folder / "dynamics.pt")
    return model


def ols(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 or float(np.ptp(x)) == 0:
        return np.nan, np.full(len(x), np.nan)
    design = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]; fitted = design @ beta
    return -float(beta[1]), y - fitted


def scalar_r2(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[finite], prediction[finite]
    sst = np.square(truth - truth.mean()).sum()
    return float(1 - np.square(truth - prediction).sum() / sst) if len(truth) > 1 and sst > 0 else np.nan


@torch.no_grad()
def sample_paths(model, row, stats, samples=8):
    z = row["z"]; paths = z.new_empty((samples, 200, 2)); paths[:, :2] = z[:2]
    current = z[1].expand(samples, -1).clone(); previous = (z[1] - z[0]).expand(samples, -1).clone()
    context = ((row["context"] - stats["context_mean"]) / stats["context_std"]).expand(samples, -1)
    source_offset = 0 if row["source"] == "node_mixed" else 100000
    generator = torch.Generator().manual_seed(91823 + source_offset + int(row["trajectory_id"]))
    for frame in range(1, 199):
        current_norm = (current - stats["z_mean"]) / stats["z_std"]
        previous_norm = (previous - stats["dz_mean"]) / stats["dz_std"]
        progress = current.new_full((samples,), frame / 199)
        mean, log_sd = model.distribution(current_norm, previous_norm, context, progress)
        normalized_delta = mean + log_sd.exp() * torch.randn((samples, 2), generator=generator)
        delta = normalized_delta * stats["dz_std"] + stats["dz_mean"]
        current = current + delta; previous = delta; paths[:, frame + 1] = current
    return paths


@torch.no_grad()
def evaluate(result, model, stats, records, folder):
    rows = []
    for row in records:
        simulation = row["simulation"]; reference = simulation[0]; pref, _ = _pref(reference)
        sides = directional_side_indices_from_box(pref, quantile=.10); xy = pref.x[:, :2].numpy()
        width0 = float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean())
        height0 = float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean())
        def wh(graph):
            q = physical(graph, reference).x[:, :2].numpy()
            return float(q[sides["right"], 0].mean() - q[sides["left"], 0].mean()), float(q[sides["top"], 1].mean() - q[sides["bottom"], 1].mean())
        truth = [(width0, height0) if frame == 0 else wh(simulation[frame]) for frame in range(200)]
        paths = sample_paths(model, row, stats)
        predicted = np.empty((len(paths), 200, 2)); predicted[:, 0] = (width0, height0); predicted[:, 1] = truth[1]
        reconstructed = np.empty((200, 2)); reconstructed[:2] = truth[:2]
        for frame in range(2, 200):
            graph = decode_latent_to_graph(result["ae"], simulation, row["z"][frame], frame, pos_dim=2,
                    ae_target_mode=result["params"]["ae_target_mode"], normalizers=result["normalizers"], device="cpu")
            reconstructed[frame] = wh(graph)
        for sample in range(len(paths)):
            for frame in range(2, 200):
                graph = decode_latent_to_graph(result["ae"], simulation, paths[sample, frame], frame, pos_dim=2,
                        ae_target_mode=result["params"]["ae_target_mode"], normalizers=result["normalizers"], device="cpu")
                predicted[sample, frame] = wh(graph)
        truth = np.asarray(truth); true_x = (truth[:, 0] - width0) / width0; true_y = (truth[:, 1] - height0) / height0
        true_p, true_residual = ols(true_x[30:], true_y[30:])
        ae_x = (reconstructed[:, 0] - width0) / width0; ae_y = (reconstructed[:, 1] - height0) / height0
        ae_p, ae_residual = ols(ae_x[30:], ae_y[30:])
        sample_p, sample_sd = [], []
        for sample in predicted:
            px = (sample[:, 0] - width0) / width0; py = (sample[:, 1] - height0) / height0
            value, residual = ols(px[30:], py[30:]); sample_p.append(value); sample_sd.append(float(np.std(residual, ddof=1)))
        mean_series = predicted.mean(0); mean_x = (mean_series[:, 0] - width0) / width0; mean_y = (mean_series[:, 1] - height0) / height0
        mean_p, _ = ols(mean_x[30:], mean_y[30:])
        true_dz = np.diff(row["z"].numpy(), axis=0)
        predicted_dz = np.diff(paths.numpy(), axis=1)
        rows.append({"source": row["source"], "trajectory_id": row["trajectory_id"], "temperature_K": row["temperature_K"],
                     "true_p_ratio": true_p, "ensemble_mean_p_ratio": mean_p,
                     "ae_reconstruction_p_ratio": ae_p,
                     "sample_p_ratio_mean": float(np.mean(sample_p)), "sample_p_ratio_sd": float(np.std(sample_p, ddof=1)),
                     "p_ratio_interval_contains_truth": bool(np.quantile(sample_p, .05) <= true_p <= np.quantile(sample_p, .95)) if np.isfinite(true_p) and np.isfinite(sample_p).all() else np.nan,
                     "true_transverse_residual_sd": float(np.std(true_residual, ddof=1)),
                     "ae_transverse_residual_sd": float(np.std(ae_residual, ddof=1)),
                     "pred_transverse_residual_sd_mean": float(np.mean(sample_sd)),
                     "pred_transverse_residual_sd_sd": float(np.std(sample_sd, ddof=1)),
                     "true_latent_increment_jitter": float(np.sqrt(np.mean(np.square(true_dz - true_dz.mean(0))))),
                     "pred_latent_increment_jitter_mean": float(np.mean(np.sqrt(np.mean(np.square(predicted_dz - predicted_dz.mean(1, keepdims=True)), axis=(1, 2))))),
                     "observed_frames": 2, "forecast_frames": 198, "samples": len(paths)})
        print("evaluated", row["source"], row["trajectory_id"], row["temperature_K"], flush=True)
    table = pd.DataFrame(rows); table.to_csv(folder / "per_network.csv", index=False)
    def summarize(group, source, temperature):
        valid_ae = np.isfinite(group.true_p_ratio) & np.isfinite(group.ae_reconstruction_p_ratio)
        valid_rollout = np.isfinite(group.true_p_ratio) & np.isfinite(group.ensemble_mean_p_ratio)
        return {"source": source, "temperature_K": temperature, "total": len(group),
                          "ae_valid": int(valid_ae.sum()), "rollout_valid": int(valid_rollout.sum()),
                          "ae_p_ratio_r2": scalar_r2(group.true_p_ratio, group.ae_reconstruction_p_ratio),
                          "ae_p_ratio_mae": float(np.abs(group.loc[valid_ae, "true_p_ratio"] - group.loc[valid_ae, "ae_reconstruction_p_ratio"]).mean()),
                          "p_ratio_r2": scalar_r2(group.true_p_ratio, group.ensemble_mean_p_ratio),
                          "p_ratio_mae": float(np.abs(group.loc[valid_rollout, "true_p_ratio"] - group.loc[valid_rollout, "ensemble_mean_p_ratio"]).mean()),
                          "p_ratio_interval_coverage": float(group.p_ratio_interval_contains_truth.mean()),
                          "true_residual_sd_mean": float(group.true_transverse_residual_sd.mean()),
                          "pred_residual_sd_mean": float(group.pred_transverse_residual_sd_mean.mean()),
                          "ae_residual_sd_ratio": float(group.ae_transverse_residual_sd.mean() / group.true_transverse_residual_sd.mean()),
                          "residual_sd_ratio": float(group.pred_transverse_residual_sd_mean.mean() / group.true_transverse_residual_sd.mean()),
                          "latent_jitter_ratio": float(group.pred_latent_increment_jitter_mean.mean() / group.true_latent_increment_jitter.mean())}
    summaries = [summarize(group, keys[0], keys[1])
                 for keys, group in table.groupby(["source", "temperature_K"])]
    pd.DataFrame(summaries).to_csv(folder / "summary_by_temperature.csv", index=False)
    pd.DataFrame([summarize(group, source, "all_high_T")
                  for source, group in table.groupby("source")]).to_csv(folder / "summary_by_source.csv", index=False)


def stage_ae(smoke):
    p = protocol(); folder = OUT / (("ae_smoke" if smoke else "ae") + os.environ.get("LSS_RUN_TAG", ""))
    folder.mkdir(exist_ok=False); ids_train = p["train_ids"][:1] if smoke else p["train_ids"]; ids_val = p["validation_ids"][:1] if smoke else p["validation_ids"]
    source = source_spec(ids_train, ids_val); cfg = ae_config(folder); seed_everything(123)
    dump(folder / "planned_recipe.json", {"status": "running", "config": cfg, "source": source, "seed": 123,
         "job_id": os.environ.get("PBS_JOBID"), "protocol_sha256": sha(CODE / "protocol.json"),
         "selection": "coordinate validation MSE only", "response_selection": False})
    result = run_latent_experiment(source, cfg, device="cpu")
    expected_train_temperatures = [0.1] if smoke else [0.1, 1.0]
    assert check_temperatures(result["train_data"], p["train_temperatures_K"]) == expected_train_temperatures
    check_temperatures(result["val_data"], p["train_temperatures_K"])
    result["ae_history"].to_csv(folder / "ae_warmup_history.csv", index=False)
    ae_sha = base_train.fit_uniform(result, folder, 1 if smoke else 60)
    dump(folder / "recipe.json", {"status": "completed", "config": cfg, "source": source, "seed": 123,
         "job_id": os.environ.get("PBS_JOBID"), "ae_sha256": ae_sha, "protocol_sha256": sha(CODE / "protocol.json"),
         "selection": "coordinate validation MSE only", "response_selection": False})
    dump(folder / "completed.json", {"status": "completed", "ae_sha256": ae_sha, "job_id": os.environ.get("PBS_JOBID")})


def stage_dynamics(smoke):
    tag = os.environ.get("LSS_RUN_TAG", "")
    p = protocol(); ae_folder = OUT / (("ae_smoke" if smoke else "ae") + tag); folder = OUT / (("dynamics_smoke" if smoke else "dynamics") + tag)
    folder.mkdir(exist_ok=False); result = load_ae(ae_folder)
    ids_train = p["train_ids"][:1] if smoke else p["train_ids"]; ids_val = p["validation_ids"][:1] if smoke else p["validation_ids"]
    source = source_spec(ids_train, ids_val); train, val, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    check_temperatures(train, p["train_temperatures_K"]); check_temperatures(val, p["train_temperatures_K"])
    train_records = encode_records(result, train, ids_train * len(SOURCES)); val_records = encode_records(result, val, ids_val * len(SOURCES))
    stats = normalization(train_records); model = fit_dynamics(train_records, val_records, stats, folder, smoke)
    evaluation_ids = p["evaluation_ids"][:1] if smoke else p["evaluation_ids"]
    manifest = json.loads((OUT / "data_manifest.json").read_text()); evaluation_records = []
    for name in SOURCES:
        path = manifest["datasets"][name]["test_path"]
        spec = {"name": name, "label": name, "path": path, "train_count": 0, "val_count": len(evaluation_ids),
                "split_indices": {"train": [], "val": evaluation_ids, "test": []}}
        evaluation_source = {"name": name, "label": name, "path": path, "dataset_name": name,
                "source_name": name, "coordinate_normalization": "position_normalization", "dataset_mixture": [spec]}
        _, simulations, _, _ = resolve_train_val_test(evaluation_source, result["params"], split_seed=123)
        assert set(check_temperatures(simulations, p["evaluation_temperatures_K"])) <= set(p["evaluation_temperatures_K"])
        evaluation_records += encode_records(result, simulations, evaluation_ids)
    evaluate(result, model, stats, evaluation_records, folder)
    dump(folder / "completed.json", {"status": "completed", "job_id": os.environ.get("PBS_JOBID"),
         "ae_sha256": sha(ae_folder / "ae.pt"), "dynamics_sha256": sha(folder / "dynamics.pt"),
         "observed_frames": [0, 1], "response_selection": False, "final_test_used": False})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("ae", "dynamics", "assert"), required=True); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    p = protocol(); assert len(p["train_ids"]) == 29 and len(p["validation_ids"]) == 10 and len(p["evaluation_ids"]) == 81
    assert not (set(p["train_ids"]) | set(p["validation_ids"])) & set(p["evaluation_ids"])
    assert max(set(p["train_ids"]) | set(p["validation_ids"]) | set(p["evaluation_ids"])) < 120
    if args.stage == "assert": print("static split assertions passed", flush=True)
    elif args.stage == "ae": stage_ae(args.smoke)
    else: stage_dynamics(args.smoke)


if __name__ == "__main__": main()
