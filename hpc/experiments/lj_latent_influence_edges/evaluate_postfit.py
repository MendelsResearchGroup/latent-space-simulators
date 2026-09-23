#!/usr/bin/env python3
"""Frozen-checkpoint post-fit response evaluation for the three edge arms."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "notebooks/results/lj_latent_influence_edges/code_v2"
OUT = ROOT / "notebooks/results/lj_latent_influence_edges"
os.environ["LSS_PROJECT_ROOT"] = str(ROOT)
os.environ["LSS_CODE_ROOT"] = str(FROZEN)

spec = importlib.util.spec_from_file_location("frozen_latent_edge_run", FROZEN / "run.py")
assert spec is not None and spec.loader is not None
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fitted_response(path: torch.Tensor, sides: dict, horizon: int) -> dict[str, float]:
    frames = np.arange(horizon + 1, dtype=float)
    values = path[:horizon + 1].detach().cpu().double()
    width = values[:, sides["right"], 0].mean(1) - values[:, sides["left"], 0].mean(1)
    height = values[:, sides["top"], 1].mean(1) - values[:, sides["bottom"], 1].mean(1)
    strain_x = ((width - width[0]) / width[0]).numpy()
    strain_y = ((height - height[0]) / height[0]).numpy()
    design = np.column_stack((np.ones(len(frames)), frames))
    fit_x = design @ np.linalg.lstsq(design, strain_x, rcond=None)[0]
    fit_y = design @ np.linalg.lstsq(design, strain_y, rcond=None)[0]
    slope_x = float(np.linalg.lstsq(design, strain_x, rcond=None)[0][1])
    slope_y = float(np.linalg.lstsq(design, strain_y, rcond=None)[0][1])
    axial = slope_x * horizon
    valid = np.isfinite(slope_x) and np.isfinite(slope_y) and abs(axial) >= 1e-3
    def fit_r2(observed, fitted):
        denominator = np.square(observed - observed.mean()).sum()
        return float(1.0 - np.square(observed - fitted).sum() / denominator) if denominator > 0 else float("nan")
    return {"p_ratio": float(-slope_y / slope_x) if valid else float("nan"),
            "valid": bool(valid), "slope_x": slope_x, "slope_y": slope_y,
            "fitted_axial_total_strain": axial,
            "fit_r2_x": fit_r2(strain_x, fit_x), "fit_r2_y": fit_r2(strain_y, fit_y)}


def response_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.square(truth - truth.mean()).sum()
    return float(1.0 - np.square(truth - prediction).sum() / denominator) if denominator > 0 else float("nan")


def main() -> None:
    protocol = json.loads((FROZEN / "protocol.json").read_text())
    manifest = json.loads((FROZEN / "manifest.json").read_text())
    assert {name: sha(FROZEN / name) for name in manifest} == manifest
    data = torch.load(ROOT / protocol["frozen_inputs"]["cache"], map_location="cpu", weights_only=False)
    ae = run.load_ae(data)
    latent_model = run.make_trajectory_propagator("history_attention", "mean")
    latent_model.load_state_dict(torch.load(ROOT / protocol["frozen_inputs"]["latent_propagator"],
                                             map_location="cpu", weights_only=False)["state_dict"])
    latent_model.eval()
    rows, edge_rows = [], []
    for arm in ("base", "random", "latent"):
        _, validation = run.prepare_records(data, ae, latent_model, arm, False)
        for record in validation:
            start = record["base_edge_count"]
            extra = record["edge_index"][:, start:]
            reference_positions = record["positions"][0]
            for source, target in extra.t().tolist():
                edge_rows.append({"arm": arm, "split": "validation", "original_index": record["original_index"],
                                  "source_node": source, "target_node": target,
                                  "reference_distance": float(torch.linalg.vector_norm(reference_positions[target] - reference_positions[source]))})
        for seed in (123, 456, 786):
            checkpoint_path = OUT / "runs" / f"{arm}_s{seed}" / "model.pt"
            completed = OUT / "runs" / f"{arm}_s{seed}" / "completed.json"
            assert completed.exists() and json.loads(completed.read_text())["status"] == "completed"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = run.NormalizedAccelerationGNN(hidden_dim=64, steps=4)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            predicted = run.batched_rollout(model, validation, checkpoint["normalizers"])
            for record, prediction in zip(validation, predicted):
                reference = record["row"]["physical_reference"]
                sides = run.directional_side_indices_from_box(reference, quantile=0.10)
                truth_physical = record["row"]["physical_positions"]
                prediction_physical = run.physical_positions(record, prediction)
                for horizon in (50, 100):
                    truth = fitted_response(truth_physical, sides, horizon)
                    estimate = fitted_response(prediction_physical, sides, horizon)
                    rows.append({"arm": arm, "seed": seed, "split": "validation",
                                 "original_index": record["original_index"], "frame": horizon,
                                 "true_p_ratio": truth["p_ratio"], "pred_p_ratio": estimate["p_ratio"],
                                 "finite_pair": truth["valid"] and estimate["valid"],
                                 "true_fit_r2_x": truth["fit_r2_x"], "true_fit_r2_y": truth["fit_r2_y"],
                                 "pred_fit_r2_x": estimate["fit_r2_x"], "pred_fit_r2_y": estimate["fit_r2_y"],
                                 "true_fitted_axial_strain": truth["fitted_axial_total_strain"],
                                 "pred_fitted_axial_strain": estimate["fitted_axial_total_strain"],
                                 "coordinate_mse": float((prediction_physical[horizon] - truth_physical[horizon]).square().mean()),
                                 "p_ratio_method": "physical_strain_time_ols_intercept_frames0_to_horizon_fixed_sides_q0.10_min_axial1e-3"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "postfit_per_network.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(OUT / "validation_added_edges.csv", index=False)
    summary = []
    for (arm, seed, frame), group in table.groupby(["arm", "seed", "frame"], sort=False):
        valid = group[group.finite_pair]
        summary.append({"arm": arm, "seed": seed, "split": "validation", "frame": frame,
                        "postfit_p_ratio_r2": response_r2(valid.true_p_ratio.to_numpy(), valid.pred_p_ratio.to_numpy()),
                        "p_ratio_valid": len(valid), "p_ratio_total": len(group),
                        "coordinate_mse": float(group.coordinate_mse.mean()),
                        "true_fit_r2_x_median": float(valid.true_fit_r2_x.median()),
                        "true_fit_r2_y_median": float(valid.true_fit_r2_y.median()),
                        "pred_fit_r2_x_median": float(valid.pred_fit_r2_x.median()),
                        "pred_fit_r2_y_median": float(valid.pred_fit_r2_y.median())})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "postfit_summary.csv", index=False)
    aggregate = summary.groupby(["arm", "frame"], as_index=False).agg(
        postfit_p_ratio_r2_mean=("postfit_p_ratio_r2", "mean"),
        postfit_p_ratio_r2_sd=("postfit_p_ratio_r2", "std"),
        coordinate_mse_mean=("coordinate_mse", "mean"),
        coordinate_mse_sd=("coordinate_mse", "std"),
        p_ratio_valid=("p_ratio_valid", "sum"),
        p_ratio_total=("p_ratio_total", "sum"),
    )
    aggregate.to_csv(OUT / "postfit_seed_summary.csv", index=False)
    metadata = {"status": "completed", "script_sha256": sha(Path(__file__)),
                "frozen_manifest_sha256": sha(FROZEN / "manifest.json"), "final_test_used": False,
                "response_used_for_training_or_selection": False,
                "metric": "two time-OLS strain fits with intercept over frames0..horizon; p=-slope_y/slope_x"}
    (OUT / "postfit_evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nThree-seed summary\n", aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
