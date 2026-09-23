#!/usr/bin/env python3
"""Frozen-AE reconstruction evaluation for the matched LJ scope-20 study.

This intentionally evaluates only the common held-out noisy-LJ validation
trajectories.  It neither trains nor selects a checkpoint, and it uses the
same fixed boundary-node groups for truth and decoded coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
CODE = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2"
OUT = ROOT / "notebooks/results/lj_mp2_scope20/evaluation"
sys.path.insert(0, str(CODE / "src"))

from graph_utils import directional_side_indices_from_box
from lss.latent.experiment import _load_ae_cache
from lss.latent.training import decode_latent_to_graph, encode_frame_latent


SCOPES = ("ljonly60", "shared3x20")
SEEDS = (123, 456, 786)
HORIZONS = (10, 20, 30, 40, 50, 75, 100, 150, 199)
METHOD = "physical_strain_time_ols_intercept_frames0_to_horizon_fixed_sides_q0.10_min_axial1e-3"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_positions(graph, *, normalization_graph=None) -> torch.Tensor:
    """Invert the per-reference-box coordinate normalization used by the AE."""
    metadata = graph if normalization_graph is None else normalization_graph
    center = metadata.reference_box_center.to(dtype=torch.float64)
    half = metadata.reference_box_half_extent.to(dtype=torch.float64)
    return graph.x[:, :2].detach().cpu().to(dtype=torch.float64) * half + center


def fitted_response(path: torch.Tensor, sides: dict, horizon: int) -> dict:
    """Fit physical x/y boundary strains over frames 0..horizon by OLS."""
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

    def r2(values: np.ndarray, fitted: np.ndarray) -> float:
        denominator = np.square(values - values.mean()).sum()
        return float(1.0 - np.square(values - fitted).sum() / denominator) if denominator > 0 else float("nan")

    valid = bool(np.isfinite(coef_x).all() and np.isfinite(coef_y).all() and abs(axial) >= 1e-3)
    return {
        "p_ratio": float(-coef_y[1] / coef_x[1]) if valid else float("nan"),
        "valid": valid,
        "slope_x": float(coef_x[1]), "slope_y": float(coef_y[1]),
        "fitted_axial_total_strain": axial,
        "fit_r2_x": r2(strain_x, fitted_x), "fit_r2_y": r2(strain_y, fitted_y),
    }


def response_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.square(truth - truth.mean()).sum()
    return float(1.0 - np.square(truth - prediction).sum() / denominator) if denominator > 0 else float("nan")


def checkpoint_paths() -> list[Path]:
    return [ROOT / "notebooks/results/lj_mp2_scope20" / f"{scope}_s{seed}" / "full" / "ae.pt"
            for scope in SCOPES for seed in SEEDS]


def reconstruct_path(result: dict, sim, max_horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    params, ae, normalizers = result["params"], result["ae"], result["normalizers"]
    truth = [physical_positions(sim[0])]
    decoded = [physical_positions(sim[0])]
    with torch.no_grad():
        for frame in range(1, max_horizon + 1):
            z = encode_frame_latent(
                ae, sim, frame, pos_dim=int(params["pos_dim"]),
                node_feature_mode=params["node_feature_mode"], normalizers=normalizers,
                device="cpu", edge_mode=getattr(ae, "edge_mode", None),
            )
            prediction = decode_latent_to_graph(
                ae, sim, z, frame, pos_dim=int(params["pos_dim"]),
                ae_target_mode=params["ae_target_mode"], normalizers=normalizers, device="cpu",
            )
            truth.append(physical_positions(sim[frame]))
            # ``decode_latent_to_graph`` deliberately returns only the static
            # model graph plus decoded coordinates, so invert its coordinates
            # with the original trajectory's frozen reference-box metadata.
            decoded.append(physical_positions(prediction, normalization_graph=sim[0]))
    return torch.stack(truth), torch.stack(decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate the frozen-run contract without loading checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = checkpoint_paths()
    missing = [str(path) for path in paths if not path.exists()]
    if args.dry_run:
        print(json.dumps({"required_checkpoints": len(paths), "existing": len(paths) - len(missing),
                          "missing": missing, "horizons": HORIZONS, "final_test_used": False}, indent=2))
        return
    if missing:
        raise FileNotFoundError("All six frozen checkpoints are required before evaluation:\n" + "\n".join(missing))
    if OUT.exists() and any(OUT.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{OUT} already contains results; use --overwrite only for an intentional rerun.")
    OUT.mkdir(parents=True, exist_ok=True)

    rows, checkpoint_hashes, validation_fingerprints = [], {}, []
    common_truth = None
    for scope in SCOPES:
        for seed in SEEDS:
            path = ROOT / "notebooks/results/lj_mp2_scope20" / f"{scope}_s{seed}" / "full" / "ae.pt"
            checkpoint_hashes[f"{scope}_s{seed}"] = sha(path)
            result = _load_ae_cache(path, {}, device="cpu")
            validation = result["val_data"]
            if len(validation) != 20 or any(str(getattr(sim[0], "source_name", "")) != "lj_noisy" for sim in validation):
                raise RuntimeError(f"{scope} seed {seed}: expected exactly 20 held-out noisy-LJ simulations")
            truth_fingerprint = hashlib.sha256()
            for sim_idx, sim in enumerate(validation):
                if len(sim) <= max(HORIZONS):
                    raise RuntimeError(f"validation trajectory {sim_idx} lacks frame {max(HORIZONS)}")
                truth_path, decoded_path = reconstruct_path(result, sim, max(HORIZONS))
                truth_fingerprint.update(truth_path.numpy().tobytes())
                sides = directional_side_indices_from_box(sim[0], quantile=0.10)
                for horizon in HORIZONS:
                    truth = fitted_response(truth_path, sides, horizon)
                    prediction = fitted_response(decoded_path, sides, horizon)
                    rows.append({
                        "scope": scope, "seed": seed, "split": "validation", "source": "lj_noisy",
                        "sim_idx": sim_idx, "frame": horizon,
                        "true_p_ratio": truth["p_ratio"], "pred_p_ratio": prediction["p_ratio"],
                        "finite_pair": truth["valid"] and prediction["valid"],
                        "true_slope_x": truth["slope_x"], "true_slope_y": truth["slope_y"],
                        "pred_slope_x": prediction["slope_x"], "pred_slope_y": prediction["slope_y"],
                        "true_fitted_axial_strain": truth["fitted_axial_total_strain"],
                        "pred_fitted_axial_strain": prediction["fitted_axial_total_strain"],
                        "true_fit_r2_x": truth["fit_r2_x"], "true_fit_r2_y": truth["fit_r2_y"],
                        "pred_fit_r2_x": prediction["fit_r2_x"], "pred_fit_r2_y": prediction["fit_r2_y"],
                        "coordinate_mse": float((decoded_path[horizon] - truth_path[horizon]).square().mean()),
                        "p_ratio_method": METHOD,
                    })
            fingerprint = truth_fingerprint.hexdigest()
            validation_fingerprints.append({"scope": scope, "seed": seed, "physical_truth_sha256": fingerprint})
            if common_truth is None:
                common_truth = fingerprint
            elif fingerprint != common_truth:
                raise RuntimeError("The validation physical trajectories differ across matched arms/seeds")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "reconstruction_postfit_per_network.csv", index=False)
    seed_rows = []
    for (scope, seed, frame), group in table.groupby(["scope", "seed", "frame"], sort=True):
        valid = group[group.finite_pair]
        seed_rows.append({"scope": scope, "seed": seed, "split": "validation", "source": "lj_noisy", "frame": frame,
                          "postfit_p_ratio_r2": response_r2(valid.true_p_ratio.to_numpy(), valid.pred_p_ratio.to_numpy()),
                          "p_ratio_valid": len(valid), "p_ratio_total": len(group),
                          "coordinate_mse": float(group.coordinate_mse.mean())})
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(OUT / "reconstruction_postfit_seed_summary.csv", index=False)
    summary = seed_summary.groupby(["scope", "split", "source", "frame"], as_index=False).agg(
        seeds=("seed", "nunique"), postfit_p_ratio_r2_mean=("postfit_p_ratio_r2", "mean"),
        postfit_p_ratio_r2_sd=("postfit_p_ratio_r2", "std"), coordinate_mse_mean=("coordinate_mse", "mean"),
        coordinate_mse_sd=("coordinate_mse", "std"), p_ratio_valid=("p_ratio_valid", "sum"),
        p_ratio_total=("p_ratio_total", "sum"))
    summary.to_csv(OUT / "reconstruction_postfit_summary.csv", index=False)
    (OUT / "evaluation.json").write_text(json.dumps({
        "status": "completed", "script_sha256": sha(Path(__file__)), "checkpoint_sha256": checkpoint_hashes,
        "validation_physical_fingerprints": validation_fingerprints, "horizons": HORIZONS,
        "metric": METHOD, "final_test_used": False, "response_used_for_training_or_selection": False,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
