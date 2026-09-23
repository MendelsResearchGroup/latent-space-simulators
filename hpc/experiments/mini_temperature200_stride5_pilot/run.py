#!/usr/bin/env python3
"""Train and evaluate a five-stored-frame latent propagator."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import directional_side_indices_from_box

from lss.dynamics.experiment import resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.training import (
    decode_latent_to_graph,
    encode_frame_latent,
    encode_reference_context,
    latent_step,
)


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/mini_temperature200_frames500"
BASELINE = BASE / "all4_twenty_each"
OUT = ROOT / "notebooks/results/mini_temperature200_stride5_pilot"
CODE = Path(__file__).resolve().parent
SOURCES = ("node_low", "node_mixed", "stiffness_low", "stiffness_mixed")
TARGET_FRAMES = tuple(range(0, 500, 5))
MIXED_FIT_FRAMES = tuple(range(150, 500, 5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_physical_module():
    path = ROOT / "notebooks/results/all4_coverage_1000/code_v5/evaluate.py"
    spec = importlib.util.spec_from_file_location("stride5_physical", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def r2(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    good = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[good], prediction[good]
    denominator = float(np.square(truth - truth.mean()).sum()) if len(truth) else 0.0
    return float(1.0 - np.square(truth - prediction).sum() / denominator) if len(truth) >= 2 and denominator > 0 else float("nan")


def ols_p(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 8 or float(np.square(x - x.mean()).sum()) <= 0:
        return float("nan"), float("nan")
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(-beta[1]), r2(y, design @ beta)


def source_for_smoke(source: dict) -> dict:
    value = copy.deepcopy(source)
    for component in value["dataset_mixture"]:
        component["train_count"] = 1
        component["val_count"] = 1
        component["split_indices"] = {"train": [0], "val": [20], "test": []}
    return value


def make_config(folder: Path, seed: int, smoke: bool) -> dict:
    config = copy.deepcopy(json.loads((BASELINE / "propagator_recipe.json").read_text())["config"])
    config.update(
        cache_path=str(folder / "propagator.pt"),
        pretrained_ae_cache_path=str(BASELINE / "ae.pt"),
        model_seed=int(seed),
        frame_skip=5,
        ae_train_frame_skip=1,
        ae_val_frame_skip=1,
        dyn_max_train_transitions_per_sim=99,
        dyn_max_val_transitions_per_sim=99,
        rollout_steps_grid=[99],
        dyn_max_epochs=1 if smoke else 60,
        dyn_patience=1 if smoke else 12,
        force_train=True,
        propagator_context_include_temperature=False,
        propagator_context_include_source_id=False,
        propagator_position_loss_weight=0.0,
        physics_loss_enabled=False,
    )
    return config


def heldout_simulations(result: dict, source: dict, source_name: str, smoke: bool) -> list:
    component = next(item for item in source["dataset_mixture"] if item["name"] == source_name)
    ids = [40] if smoke else list(range(40, 120))
    selected = copy.deepcopy(component)
    selected["train_count"] = 0
    selected["val_count"] = len(ids)
    selected["split_indices"] = {"train": [], "val": ids, "test": []}
    spec = {
        "name": source_name,
        "label": source_name,
        "path": selected["path"],
        "dataset_name": source_name,
        "source_name": source_name,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [selected],
    }
    _, simulations, _, _ = resolve_train_val_test(spec, result["params"], split_seed=123)
    if [int(sim[0].source_sim_id) for sim in simulations] != ids:
        raise RuntimeError(f"Heldout identities changed for {source_name}")
    return simulations


@torch.no_grad()
def evaluate(result: dict, source: dict, seed: int, smoke: bool, folder: Path) -> None:
    physical_module = load_physical_module()
    rows = []
    model = result["ae"].eval()
    dynamics = result["dyn"].eval()
    params = result["params"]
    normalizers = result["normalizers"]
    for source_name in SOURCES:
        simulations = heldout_simulations(result, source, source_name, smoke)
        for simulation in simulations:
            reference = simulation[0]
            physical_reference, _ = physical_module._pref(reference)
            sides = directional_side_indices_from_box(physical_reference, quantile=0.10)
            reference_xy = physical_reference.x[:, :2].detach().cpu().numpy()
            width0 = float(reference_xy[sides["right"], 0].mean() - reference_xy[sides["left"], 0].mean())
            height0 = float(reference_xy[sides["top"], 1].mean() - reference_xy[sides["bottom"], 1].mean())

            def strains(graph) -> tuple[float, float]:
                xy = physical_module.physical(graph, reference).x[:, :2].detach().cpu().numpy()
                width = float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean())
                height = float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean())
                return (width - width0) / width0, (height - height0) / height0

            z = encode_frame_latent(
                model, simulation, 0, pos_dim=2,
                node_feature_mode=params["node_feature_mode"],
                normalizers=normalizers, device="cpu",
            )
            context = encode_reference_context(
                model, simulation, pos_dim=2, normalizers=normalizers,
                device="cpu", include_temperature=False,
                include_source_id=False, pool_mode="mean",
            )
            trajectory_rows = [{"frame": 0, "true_x": 0.0, "true_y": 0.0, "pred_x": 0.0, "pred_y": 0.0}]
            for frame in TARGET_FRAMES[1:]:
                z = latent_step(dynamics, z, result["latent_stats"], context=context, loss_mode="delta")
                prediction = decode_latent_to_graph(
                    model, simulation, z, frame, pos_dim=2,
                    ae_target_mode=params["ae_target_mode"],
                    normalizers=normalizers, device="cpu",
                )
                true_x, true_y = strains(simulation[frame])
                pred_x, pred_y = strains(prediction)
                trajectory_rows.append({"frame": frame, "true_x": true_x, "true_y": true_y, "pred_x": pred_x, "pred_y": pred_y})
            frame_table = pd.DataFrame(trajectory_rows)
            final = frame_table.iloc[-1]
            true_endpoint = -final.true_y / final.true_x if abs(final.true_x) > 1e-12 else float("nan")
            pred_endpoint = -final.pred_y / final.pred_x if abs(final.pred_x) > 1e-12 else float("nan")
            selected = frame_table[frame_table.frame.isin(MIXED_FIT_FRAMES)]
            true_fit, true_fit_r2 = ols_p(selected.true_x, selected.true_y)
            pred_fit, pred_fit_r2 = ols_p(selected.pred_x, selected.pred_y)
            rows.append({
                "seed": seed,
                "source": source_name,
                "trajectory_id": int(reference.source_sim_id),
                "temperature_K": float(reference.temperature),
                "endpoint_frame": 495,
                "true_endpoint_p_ratio": true_endpoint,
                "pred_endpoint_p_ratio": pred_endpoint,
                "true_linear_p_ratio": true_fit,
                "pred_linear_p_ratio": pred_fit,
                "true_linear_fit_r2": true_fit_r2,
                "pred_linear_fit_r2": pred_fit_r2,
            })
    table = pd.DataFrame(rows)
    table.to_csv(folder / "per_network.csv", index=False)
    summaries = []
    for source_name, group in table.groupby("source"):
        mixed = source_name.endswith("_mixed")
        truth = "true_linear_p_ratio" if mixed else "true_endpoint_p_ratio"
        prediction = "pred_linear_p_ratio" if mixed else "pred_endpoint_p_ratio"
        good = group[[truth, prediction]].replace([np.inf, -np.inf], np.nan).dropna()
        summaries.append({
            "seed": seed,
            "source": source_name,
            "metric": "strain_OLS_frames150:5:495" if mixed else "physical_endpoint_frame495",
            "p_ratio_r2": r2(good[truth], good[prediction]),
            "p_ratio_mae": float(np.abs(good[truth] - good[prediction]).mean()),
            "valid": len(good),
            "total": len(group),
        })
    pd.DataFrame(summaries).to_csv(folder / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(123, 456, 786), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    seed_everything(args.seed)
    tag = f"seed{args.seed}" + ("_smoke" if args.smoke else "")
    folder = OUT / tag
    folder.mkdir(parents=True, exist_ok=False)
    baseline_recipe = json.loads((BASELINE / "recipe.json").read_text())
    source = source_for_smoke(baseline_recipe["source"]) if args.smoke else baseline_recipe["source"]
    config = make_config(folder, args.seed, args.smoke)
    ae_before = sha256(BASELINE / "ae.pt")
    write_json(folder / "recipe.json", {
        "source": source,
        "config": config,
        "protocol_sha256": sha256(CODE / "protocol.json"),
        "baseline_ae_sha256": ae_before,
        "job_id": os.environ.get("PBS_JOBID"),
        "smoke": args.smoke,
    })
    result = run_latent_experiment(source, config, device="cpu")
    if sha256(BASELINE / "ae.pt") != ae_before:
        raise RuntimeError("Frozen baseline AE changed")
    result["dyn_history"].to_csv(folder / "propagator_history.csv", index=False)
    evaluate(result, source, args.seed, args.smoke, folder)
    completion = {
        "status": "complete",
        "seed": args.seed,
        "smoke": args.smoke,
        "job_id": os.environ.get("PBS_JOBID"),
        "baseline_ae_sha256_before_after": ae_before,
        "propagator_sha256": sha256(folder / "propagator.pt"),
        "valid_sources": SOURCES,
        "heldout_per_source": 1 if args.smoke else 80,
        "final_test_used": False,
    }
    write_json(folder / "completed.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
