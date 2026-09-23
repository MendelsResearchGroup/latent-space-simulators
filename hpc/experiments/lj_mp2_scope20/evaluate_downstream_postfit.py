#!/usr/bin/env python3
"""Post-fit physical response evaluation for frozen scope-20 downstream GNNs.

This consumes, but never changes, completed ``run_downstream.py`` artifacts.
The response estimator is deliberately the same for true and autonomous
predicted positions: fixed frame-zero q=.10 boundary sides and two physical
boundary-strain OLS fits (with intercept) over frames 0..horizon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
STUDY = ROOT / "notebooks/results/lj_mp2_scope20"
RUNS = STUDY / "downstream/runs"
OUT = STUDY / "downstream/evaluation"
SCOPES = ("ljonly60", "shared3x20")
ARMS = ("base", "random", "latent")
SEEDS = (123, 456, 786)
HORIZONS = (10, 20, 30, 40, 50, 75, 100)
METHOD = "physical_strain_time_ols_intercept_frames0_to_horizon_fixed_sides_q0.10_min_axial1e-3"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_runs() -> list[tuple[str, str, int, Path]]:
    return [(scope, arm, seed, RUNS / f"{scope}_{arm}_s{seed}")
            for scope in SCOPES for arm in ARMS for seed in SEEDS]


def fitted_response(path: torch.Tensor, sides: dict[str, list[int]], horizon: int) -> dict[str, float | bool]:
    """Physical width/height strain OLS with intercept over frames 0..horizon."""
    values = path[: horizon + 1].detach().cpu().double()
    width = values[:, sides["right"], 0].mean(1) - values[:, sides["left"], 0].mean(1)
    height = values[:, sides["top"], 1].mean(1) - values[:, sides["bottom"], 1].mean(1)
    strain_x = ((width - width[0]) / width[0]).numpy()
    strain_y = ((height - height[0]) / height[0]).numpy()
    time = np.arange(horizon + 1, dtype=float)
    design = np.column_stack((np.ones(len(time)), time))
    coef_x = np.linalg.lstsq(design, strain_x, rcond=None)[0]
    coef_y = np.linalg.lstsq(design, strain_y, rcond=None)[0]
    fitted_x, fitted_y = design @ coef_x, design @ coef_y
    axial = float(coef_x[1] * horizon)

    def fit_r2(observed: np.ndarray, fitted: np.ndarray) -> float:
        denominator = np.square(observed - observed.mean()).sum()
        return float(1.0 - np.square(observed - fitted).sum() / denominator) if denominator > 0 else float("nan")

    valid = bool(np.isfinite(coef_x).all() and np.isfinite(coef_y).all() and abs(axial) >= 1e-3 and abs(coef_x[1]) > 0.0)
    return {"p_ratio": float(-coef_y[1] / coef_x[1]) if valid else float("nan"), "valid": valid,
            "slope_x": float(coef_x[1]), "slope_y": float(coef_y[1]),
            "fitted_axial_total_strain": axial, "fit_r2_x": fit_r2(strain_x, fitted_x),
            "fit_r2_y": fit_r2(strain_y, fitted_y)}


def response_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.square(truth - truth.mean()).sum()
    return float(1.0 - np.square(truth - prediction).sum() / denominator) if denominator > 0 else float("nan")


def side_indices(reference) -> dict[str, list[int]]:
    """Use the frozen reference graph and its physical box to choose q=.10 sides."""
    sys.path.insert(0, str(ROOT / "notebooks/results/lj_latent_influence_edges/code_v2/src"))
    from graph_utils import directional_side_indices_from_box
    sides = directional_side_indices_from_box(reference, quantile=0.10)
    return {key: [int(index) for index in value] for key, value in sides.items()}


def validate_runs() -> tuple[list[tuple[str, str, int, Path]], list[str]]:
    ready, problems = [], []
    for scope, arm, seed, directory in required_runs():
        label = f"{scope}/{arm}/s{seed}"
        for name in ("completed.json", "recipe.json", "per_network_rows.csv", "predicted_position_paths.pt"):
            if not (directory / name).is_file():
                problems.append(f"{label}: missing {name}")
        if (directory / "completed.json").is_file():
            try:
                if json.loads((directory / "completed.json").read_text()).get("status") != "completed":
                    problems.append(f"{label}: completed.json status is not completed")
            except json.JSONDecodeError as error:
                problems.append(f"{label}: unreadable completed.json ({error})")
        if not any(problem.startswith(label + ":") for problem in problems):
            ready.append((scope, arm, seed, directory))
    return ready, problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report the complete required-run contract without evaluating")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing downstream/evaluation directory")
    args = parser.parse_args()
    ready, problems = validate_runs()
    contract = {"required_runs": len(required_runs()), "ready_runs": len(ready), "problems": problems,
                "horizons": HORIZONS, "metric": METHOD, "final_test_used": False}
    if args.dry_run:
        print(json.dumps(contract, indent=2))
        return
    if problems:
        raise RuntimeError("All 18 completed downstream artifacts are required:\n" + "\n".join(problems))
    if OUT.exists() and any(OUT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{OUT} already contains results; use --overwrite for an intentional re-evaluation")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rows, artifact_hashes, truth_fingerprints = [], {}, []
    for scope, arm, seed, directory in ready:
        artifacts = {name: directory / name for name in ("completed.json", "recipe.json", "per_network_rows.csv", "predicted_position_paths.pt")}
        artifact_hashes[f"{scope}_{arm}_s{seed}"] = {name: sha(path) for name, path in artifacts.items()}
        paths = torch.load(artifacts["predicted_position_paths.pt"], map_location="cpu", weights_only=False)
        if paths.get("positions_coordinate_system") != "physical" or paths.get("frames") != list(range(101)):
            raise RuntimeError(f"{directory}: expected physical frames 0..100")
        records = paths.get("records", [])
        if len(records) != 20:
            raise RuntimeError(f"{directory}: expected 20 validation records, found {len(records)}")
        counts = pd.read_csv(artifacts["per_network_rows.csv"])
        required_columns = {"original_index", "added_edge_count"}
        if not required_columns.issubset(counts.columns):
            raise RuntimeError(f"{directory}: per_network_rows.csv lacks {sorted(required_columns - set(counts.columns))}")
        added = counts.groupby("original_index")["added_edge_count"].agg(["min", "max"])
        if not (added["min"] == added["max"]).all():
            raise RuntimeError(f"{directory}: added-edge count changes over reporting rows")
        added_by_id = added["min"].to_dict()
        fingerprint = hashlib.sha256()
        seen = set()
        for record in records:
            network_id = int(record["original_index"])
            if network_id in seen or network_id not in added_by_id:
                raise RuntimeError(f"{directory}: path/edge rows do not form one-to-one validation records")
            seen.add(network_id)
            reference = record["reference"]
            truth_path, predicted_path = record["true_positions_physical"], record["predicted_positions_physical"]
            if tuple(truth_path.shape) != tuple(predicted_path.shape) or truth_path.shape[0] != 101 or truth_path.shape[-1] != 2:
                raise RuntimeError(f"{directory}: malformed physical path for network {network_id}")
            if not torch.isfinite(truth_path).all() or not torch.isfinite(predicted_path).all():
                raise RuntimeError(f"{directory}: non-finite physical positions for network {network_id}")
            fingerprint.update(network_id.to_bytes(8, "little", signed=True)); fingerprint.update(truth_path.detach().cpu().numpy().tobytes())
            sides = side_indices(reference)
            for horizon in HORIZONS:
                truth, prediction = fitted_response(truth_path, sides, horizon), fitted_response(predicted_path, sides, horizon)
                rows.append({"scope": scope, "arm": arm, "seed": seed, "split": "validation", "source": "lj_noisy",
                             "original_index": network_id, "frame": horizon, "true_p_ratio": truth["p_ratio"],
                             "pred_p_ratio": prediction["p_ratio"], "finite_pair": bool(truth["valid"] and prediction["valid"]),
                             "true_slope_x": truth["slope_x"], "true_slope_y": truth["slope_y"],
                             "pred_slope_x": prediction["slope_x"], "pred_slope_y": prediction["slope_y"],
                             "true_fitted_axial_strain": truth["fitted_axial_total_strain"],
                             "pred_fitted_axial_strain": prediction["fitted_axial_total_strain"],
                             "true_fit_r2_x": truth["fit_r2_x"], "true_fit_r2_y": truth["fit_r2_y"],
                             "pred_fit_r2_x": prediction["fit_r2_x"], "pred_fit_r2_y": prediction["fit_r2_y"],
                             "coordinate_mse": float((predicted_path[horizon].double() - truth_path[horizon].double()).square().mean()),
                             "added_edge_count": int(added_by_id[network_id]), "p_ratio_method": METHOD})
        if seen != set(added_by_id):
            raise RuntimeError(f"{directory}: extra edge-count records lack saved paths")
        truth_fingerprints.append({"scope": scope, "arm": arm, "seed": seed, "physical_truth_sha256": fingerprint.hexdigest()})

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "postfit_per_network.csv", index=False)
    seed_rows = []
    for keys, group in table.groupby(["scope", "arm", "seed", "split", "source", "frame"], sort=True):
        valid = group[group.finite_pair]
        seed_rows.append(dict(zip(("scope", "arm", "seed", "split", "source", "frame"), keys)) |
                         {"postfit_p_ratio_r2": response_r2(valid.true_p_ratio.to_numpy(), valid.pred_p_ratio.to_numpy()),
                          "p_ratio_valid": len(valid), "p_ratio_total": len(group),
                          "coordinate_mse": float(group.coordinate_mse.mean()), "added_edges_mean": float(group.added_edge_count.mean())})
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(OUT / "postfit_seed_summary.csv", index=False)
    aggregate = seed_summary.groupby(["scope", "arm", "split", "source", "frame"], as_index=False).agg(
        seeds=("seed", "nunique"), postfit_p_ratio_r2_mean=("postfit_p_ratio_r2", "mean"),
        postfit_p_ratio_r2_sd=("postfit_p_ratio_r2", "std"), coordinate_mse_mean=("coordinate_mse", "mean"),
        coordinate_mse_sd=("coordinate_mse", "std"), added_edges_mean=("added_edges_mean", "mean"),
        added_edges_sd=("added_edges_mean", "std"), p_ratio_valid=("p_ratio_valid", "sum"), p_ratio_total=("p_ratio_total", "sum"))
    aggregate.to_csv(OUT / "postfit_seed_aggregate.csv", index=False)
    edge_table = table.groupby(["scope", "arm", "seed", "original_index"], as_index=False).agg(added_edge_count=("added_edge_count", "first"))
    edge_table.to_csv(OUT / "added_edge_counts_per_network.csv", index=False)
    edge_summary = edge_table.groupby(["scope", "arm"], as_index=False).agg(seeds=("seed", "nunique"), networks=("original_index", "count"), added_edges_mean=("added_edge_count", "mean"), added_edges_sd=("added_edge_count", "std"), added_edges_min=("added_edge_count", "min"), added_edges_max=("added_edge_count", "max"))
    edge_summary.to_csv(OUT / "added_edge_count_summary.csv", index=False)
    output_hashes = {path.name: sha(path) for path in (OUT / "postfit_per_network.csv", OUT / "postfit_seed_summary.csv", OUT / "postfit_seed_aggregate.csv", OUT / "added_edge_counts_per_network.csv", OUT / "added_edge_count_summary.csv")}
    workflow_artifacts = {name: sha(ROOT / "hpc/experiments/lj_mp2_scope20" / name) for name in
                          ("evaluate_downstream_postfit.py", "evaluate_downstream_postfit.pbs", "build_augmented_gnn_notebook.py",
                           "downstream_manifest.json", "downstream_protocol.json")}
    metadata = {"status": "completed", "script_sha256": sha(Path(__file__)), "workflow_artifact_sha256": workflow_artifacts,
                "input_artifact_sha256": artifact_hashes,
                "output_artifact_sha256": output_hashes, "physical_truth_fingerprints": truth_fingerprints,
                "horizons": HORIZONS, "metric": METHOD, "fixed_boundary_sides": "reference graph q=0.10, reused for truth and prediction",
                "final_test_used": False, "response_used_for_training_or_selection": False}
    (OUT / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
