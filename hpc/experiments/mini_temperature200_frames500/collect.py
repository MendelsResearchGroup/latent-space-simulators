#!/usr/bin/env python3
"""Frozen heldout evaluation, low-T latents and mixed-T linear p-ratio fits."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_utils import directional_side_indices_from_box
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import encode_frame_latent

from evaluate import evaluate, physical, pref, pratio


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / os.environ.get("LSS_RESULT_DIR", "notebooks/results/mini_temperature200_frames500")
FRAMES = 500
TRAIN_FRAMES = int(os.environ.get("LSS_TRAIN_FRAMES", "500"))
EXPORT_ALL_LATENTS = os.environ.get("LSS_EXPORT_ALL_LATENTS", "0") == "1"
SOURCES = ("node_low", "node_mixed", "stiffness_low", "stiffness_mixed")
LOW = ("node_low", "stiffness_low")
MIXED = ("node_mixed", "stiffness_mixed")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(truth, prediction) -> float:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    denominator = np.square(truth - truth.mean()).sum()
    return float(1.0 - np.square(truth - prediction).sum() / denominator) if len(truth) >= 2 and denominator > 0 else float("nan")


def ols_p(frame_table: pd.DataFrame, truth: bool, start: int = 150, end: int = 499) -> tuple[float, float, int]:
    prefix = "true" if truth else "pred"
    selected = frame_table.loc[frame_table.frame.between(start, end), [f"{prefix}_boundary_strain_x", f"{prefix}_boundary_strain_y"]].to_numpy(float)
    good = np.isfinite(selected).all(axis=1)
    expected = end - start + 1
    if good.sum() != expected:
        return float("nan"), float("nan"), int(good.sum())
    x, y = selected[good].T
    if np.square(x - x.mean()).sum() <= 0:
        return float("nan"), float("nan"), len(x)
    fitted = np.c_[np.ones(len(x)), x] @ np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0]
    slope = np.linalg.lstsq(np.c_[np.ones(len(x)), x], y, rcond=None)[0][1]
    return float(-slope), r2(y, fitted), len(x)


@torch.no_grad()
def latent_rows(result: dict, simulations: list) -> pd.DataFrame:
    rows = []
    model, params, normalizers = result["ae"], result["params"], result["normalizers"]
    model.eval()
    for simulation in simulations:
        reference = simulation[0]
        physical_reference, _ = pref(reference)
        sides = directional_side_indices_from_box(physical_reference, quantile=0.10)
        for frame in range(FRAMES):
            z = encode_frame_latent(model, simulation, frame, pos_dim=2, node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device="cpu").detach().cpu().reshape(-1)
            graph = physical(simulation[frame], reference)
            if frame == 0:
                graph.x[:, :2] = physical_reference.x[:, :2]
            rows.append({
                "source": str(reference.source_name), "trajectory_id": int(reference.source_sim_id), "frame": frame,
                "md_step": frame * 200, "z0": float(z[0]), "z1": float(z[1]),
                "frame_p_ratio": pratio(physical_reference, graph, sides),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    folder = OUT / f"{args.arm}{args.tag}{'_smoke' if args.smoke else ''}"
    assert json.loads((folder / "fit_completed.json").read_text())["status"] == "completed"
    recipe = json.loads((folder / "propagator_recipe.json").read_text())
    result = load_experiment_bundle(folder / "propagator.pt", cfg=recipe["config"], device="cpu")
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    all_network, all_step, all_latent = [], [], []
    # Smoke both estimator branches: one low-T endpoint source and one mixed-T
    # linear-fit source.  The original smoke selected node_low only, leaving
    # the mixed-fit table empty and failing after otherwise successful fitting.
    selected_sources = ("node_low", "node_mixed") if args.smoke else SOURCES
    for name in selected_sources:
        entry = manifest["datasets"][name]
        assert sha(Path(entry["cache_path"])) == entry["cache_sha256"]
        ids = [40] if args.smoke else list(range(40, 120))
        spec = {"name": name, "label": name, "path": entry["cache_path"], "train_count": 0, "val_count": len(ids), "split_indices": {"train": [], "val": ids, "test": []}}
        source = {"name": name, "label": name, "path": entry["cache_path"], "dataset_name": name, "source_name": name, "coordinate_normalization": "position_normalization", "dataset_mixture": [spec]}
        _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
        assert [int(sim[0].source_sim_id) for sim in simulations] == ids
        assert all(len(sim) == FRAMES and all(int(graph.time) == frame * 200 for frame, graph in enumerate(sim)) for sim in simulations)
        table = evaluate({**result, "val_data": simulations}, seed=recipe["config"]["model_seed"], latent_dim=2, output=folder / "test" / name, horizon=FRAMES - 1, split="test")
        all_network.append(table)
        all_step.append(pd.read_csv(folder / "test" / name / "per_step.csv"))
        if name in LOW or EXPORT_ALL_LATENTS:
            all_latent.append(latent_rows(result, simulations))
        del simulations
        gc.collect()
    network = pd.concat(all_network, ignore_index=True)
    network.to_csv(folder / "test/per_network.csv", index=False)
    pd.concat(all_step, ignore_index=True).to_csv(folder / "test/per_step.csv", index=False)
    if all_latent:
        latent_name = "all_temperature_latents.csv" if EXPORT_ALL_LATENTS else "low_temperature_latents.csv"
        pd.concat(all_latent, ignore_index=True).to_csv(folder / "test" / latent_name, index=False)

    endpoint_frames = sorted({TRAIN_FRAMES - 1, FRAMES - 1})
    endpoint_summary = pd.concat(all_step, ignore_index=True)
    endpoint_summary = endpoint_summary[endpoint_summary.frame.isin(endpoint_frames)].copy()
    endpoint_summary.to_csv(folder / "test/endpoint_summary.csv", index=False)

    linear_rows = []
    for (source, mode, trajectory_id), group in network[network.source.isin(MIXED)].groupby(["source", "mode", "trajectory_id"]):
        true_p, true_fit_r2, true_pairs = ols_p(group, True)
        pred_p, pred_fit_r2, pred_pairs = ols_p(group, False)
        linear_rows.append({"source": source, "mode": mode, "trajectory_id": trajectory_id, "true_p_ratio": true_p, "pred_p_ratio": pred_p, "true_fit_r2": true_fit_r2, "pred_fit_r2": pred_fit_r2, "true_pairs": true_pairs, "pred_pairs": pred_pairs})
    linear = pd.DataFrame(linear_rows, columns=[
        "source", "mode", "trajectory_id", "true_p_ratio", "pred_p_ratio",
        "true_fit_r2", "pred_fit_r2", "true_pairs", "pred_pairs",
    ])
    linear.to_csv(folder / "test/mixed_linear_fit_per_network.csv", index=False)
    summary = []
    for (source, mode), group in linear.groupby(["source", "mode"]):
        good = group.dropna(subset=["true_p_ratio", "pred_p_ratio"])
        summary.append({"source": source, "mode": mode, "p_ratio_r2": r2(good.true_p_ratio, good.pred_p_ratio), "p_ratio_mae": float(np.abs(good.true_p_ratio - good.pred_p_ratio).mean()), "valid": len(good), "total": len(group), "true_fit_r2_median": float(good.true_fit_r2.median()), "pred_fit_r2_median": float(good.pred_fit_r2.median())})
    pd.DataFrame(summary).to_csv(folder / "test/mixed_linear_fit_summary.csv", index=False)
    training_window_rows = []
    for (source, mode, trajectory_id), group in network[network.source.isin(MIXED)].groupby(["source", "mode", "trajectory_id"]):
        true_p, true_fit_r2, true_pairs = ols_p(group, True, 0, TRAIN_FRAMES - 1)
        pred_p, pred_fit_r2, pred_pairs = ols_p(group, False, 0, TRAIN_FRAMES - 1)
        training_window_rows.append({"source": source, "mode": mode, "trajectory_id": trajectory_id,
            "true_p_ratio": true_p, "pred_p_ratio": pred_p, "true_fit_r2": true_fit_r2,
            "pred_fit_r2": pred_fit_r2, "true_pairs": true_pairs, "pred_pairs": pred_pairs})
    training_window = pd.DataFrame(training_window_rows)
    training_window.to_csv(folder / "test/mixed_training_window_fit_per_network.csv", index=False)
    training_window_summary = []
    for (source, mode), group in training_window.groupby(["source", "mode"]):
        good = group.dropna(subset=["true_p_ratio", "pred_p_ratio"])
        training_window_summary.append({"source": source, "mode": mode,
            "p_ratio_r2": r2(good.true_p_ratio, good.pred_p_ratio),
            "p_ratio_mae": float(np.abs(good.true_p_ratio - good.pred_p_ratio).mean()),
            "valid": len(good), "total": len(group), "fit_frames_start": 0,
            "fit_frames_end": TRAIN_FRAMES - 1, "true_fit_r2_median": float(good.true_fit_r2.median()),
            "pred_fit_r2_median": float(good.pred_fit_r2.median())})
    pd.DataFrame(training_window_summary).to_csv(folder / "test/mixed_training_window_fit_summary.csv", index=False)
    before = {"ae": sha(folder / "ae.pt"), "propagator": sha(folder / "propagator.pt")}
    completed = {"status": "completed", "job_id": os.environ.get("PBS_JOBID"), "arm": args.arm, "smoke": args.smoke, "checkpoint_sha256": before, "data_manifest_sha256": sha(OUT / "data_manifest.json"), "heldout_ids": [40] if args.smoke else list(range(40, 120)), "training_frames": [0, TRAIN_FRAMES - 1], "evaluation_frames": [0, 499], "endpoint_summary_frames": endpoint_frames, "mixed_training_window_fit_frames": [0, TRAIN_FRAMES - 1], "mixed_linear_fit_frames": [150, 499], "test_used_for_selection": False}
    (folder / "completed.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
