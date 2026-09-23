"""Post-fit mixed-T mean-response and thermal-residual diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate import physical, _pref
from graph_utils import directional_side_indices_from_box
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import (
    decode_latent_to_graph,
    encode_frame_latent,
    encode_reference_context,
    latent_step,
)
from train import ARM, OUT, SOURCE_AE, sha


MIXED_SOURCES = ("node_mixed", "stiffness_mixed")
HORIZONS = tuple(range(37, 200))


def r2(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    sst = float(np.square(truth - truth.mean()).sum())
    return float(1.0 - np.square(truth - prediction).sum() / sst) if len(truth) >= 2 and sst > 0 else float("nan")


def correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def ols(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    centered = x - x.mean()
    if float(np.square(centered).sum()) <= 0:
        return None
    design = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "fit_r2": r2(y, fitted), "residual": y - fitted}


def endpoint_p(width, height, width0, height0) -> float:
    dw, dh = width - width0, height - height0
    first, second = -dh / (dw + 1e-9), -dw / (dh + 1e-9)
    return float(first if abs(first) < abs(second) else second)


def load_sims(result, entry, name):
    ids = list(range(40, 120))
    spec = {"name": name, "label": name, "path": entry["test_path"], "train_count": 0,
            "val_count": len(ids), "split_indices": {"train": [], "val": ids, "test": []}}
    source = {"name": name, "label": name, "path": entry["test_path"], "dataset_name": name,
              "source_name": name, "coordinate_normalization": "position_normalization", "dataset_mixture": [spec]}
    _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    assert len(simulations) == 80
    for trajectory_id, simulation in zip(ids, simulations):
        for frame, graph in enumerate(simulation):
            graph.source_sim_id = trajectory_id
            assert graph.time == frame * 1000
    return simulations


@torch.no_grad()
def evaluate_source(result, simulations, source):
    ae, normalizers, params = result["ae"], result["normalizers"], result["params"]
    dynamics, stats = result["dyn"], result["latent_stats"]
    ae.eval(); dynamics.eval()
    full_rows, prefix_rows = [], []
    for simulation in simulations:
        reference = simulation[0]
        physical_reference, _ = _pref(reference)
        sides = directional_side_indices_from_box(physical_reference, quantile=0.10)
        ref_xy = physical_reference.x[:, :2].numpy()
        width0 = float(ref_xy[sides["right"], 0].mean() - ref_xy[sides["left"], 0].mean())
        height0 = float(ref_xy[sides["top"], 1].mean() - ref_xy[sides["bottom"], 1].mean())

        def width_height(graph):
            xy = physical(graph, reference).x[:, :2].numpy()
            return (float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean()),
                    float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean()))

        truth, reconstruction, rollout = [], [], []
        z = encode_frame_latent(ae, simulation, 0, pos_dim=2, node_feature_mode=params["node_feature_mode"],
                                normalizers=normalizers, device="cpu")
        context = encode_reference_context(ae, simulation, pos_dim=2, normalizers=normalizers, device="cpu",
                                           include_temperature=False, include_source_id=False, pool_mode="mean")
        for frame in range(200):
            encoded = encode_frame_latent(ae, simulation, frame, pos_dim=2,
                                          node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device="cpu")
            truth.append((width0, height0) if frame == 0 else width_height(simulation[frame]))
            reconstruction.append(width_height(decode_latent_to_graph(
                ae, simulation, encoded, frame, pos_dim=2, ae_target_mode=params["ae_target_mode"],
                normalizers=normalizers, device="cpu")))
            if frame == 0:
                rollout.append((width0, height0))
            else:
                z = latent_step(dynamics, z, stats, context=context, loss_mode="delta")
                rollout.append(width_height(decode_latent_to_graph(
                    ae, simulation, z, frame, pos_dim=2, ae_target_mode=params["ae_target_mode"],
                    normalizers=normalizers, device="cpu")))

        true_width, true_height = np.asarray(truth).T
        true_x = (true_width - width0) / width0
        true_y = (true_height - height0) / height0
        for mode, series in (("ae_reconstruction", reconstruction), ("autonomous_delta_mlp", rollout)):
            pred_width, pred_height = np.asarray(series).T
            pred_x = (pred_width - width0) / width0
            pred_y = (pred_height - height0) / height0
            for horizon in HORIZONS:
                true_fit = ols(true_x[30:horizon + 1], true_y[30:horizon + 1])
                pred_fit = ols(pred_x[30:horizon + 1], pred_y[30:horizon + 1])
                assert true_fit is not None and pred_fit is not None
                prefix_rows.append({
                    "source": source, "trajectory_id": int(reference.source_sim_id), "mode": mode,
                    "horizon": horizon, "md_step": horizon * 1000,
                    "true_p_ratio": -true_fit["slope"], "pred_p_ratio": -pred_fit["slope"],
                    "true_fit_r2": true_fit["fit_r2"], "pred_fit_r2": pred_fit["fit_r2"],
                })
            true_fit = ols(true_x[30:], true_y[30:])
            pred_fit = ols(pred_x[30:], pred_y[30:])
            assert true_fit is not None and pred_fit is not None
            true_residual = true_fit["residual"]
            pred_residual = pred_fit["residual"]
            full_rows.append({
                "source": source, "trajectory_id": int(reference.source_sim_id), "temperature_K": float(reference.temperature),
                "mode": mode, "true_p_ratio": -true_fit["slope"], "pred_p_ratio": -pred_fit["slope"],
                "true_fit_r2": true_fit["fit_r2"], "pred_fit_r2": pred_fit["fit_r2"],
                "true_transverse_residual_sd": float(np.std(true_residual, ddof=1)),
                "pred_transverse_residual_sd": float(np.std(pred_residual, ddof=1)),
                "residual_waveform_pearson": correlation(true_residual, pred_residual),
                "endpoint_true_p": endpoint_p(true_width[-1], true_height[-1], width0, height0),
                "endpoint_pred_p": endpoint_p(pred_width[-1], pred_height[-1], width0, height0),
            })
    return full_rows, prefix_rows


def main() -> None:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    folder = OUT / ARM
    receipt = json.loads((folder / "evaluation_completed.json").read_text())
    assert receipt["status"] == "completed" and not receipt["smoke"]
    recipe = json.loads((folder / "recipe.json").read_text())
    before = {"ae": sha(SOURCE_AE), "propagator": sha(folder / "propagator.pt")}
    result = load_experiment_bundle(folder / "propagator.pt", cfg=recipe["config"], device="cpu")
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    rows, prefixes = [], []
    for source in MIXED_SOURCES:
        source_rows, source_prefixes = evaluate_source(result, load_sims(result, manifest["datasets"][source], source), source)
        rows += source_rows; prefixes += source_prefixes
    table, prefix = pd.DataFrame(rows), pd.DataFrame(prefixes)
    analysis = folder / "postfit"
    analysis.mkdir(exist_ok=False)
    table.to_csv(analysis / "per_network.csv", index=False)
    prefix.to_csv(analysis / "per_network_prefix.csv", index=False)
    prefix_summary = []
    for (source, mode, horizon), group in prefix.groupby(["source", "mode", "horizon"]):
        prefix_summary.append({
            "source": source, "mode": mode, "horizon": horizon, "md_step": horizon * 1000,
            "p_ratio_r2": r2(group.true_p_ratio, group.pred_p_ratio),
            "p_ratio_mae": float(np.abs(group.true_p_ratio - group.pred_p_ratio).mean()),
            "valid": len(group), "total": len(group),
        })
    pd.DataFrame(prefix_summary).to_csv(analysis / "prefix_summary.csv", index=False)
    summary = []
    for (source, mode), group in table.groupby(["source", "mode"]):
        amplitude_ratio = group.pred_transverse_residual_sd / group.true_transverse_residual_sd
        summary.append({
            "source": source, "mode": mode,
            "linear_p_ratio_r2": r2(group.true_p_ratio, group.pred_p_ratio),
            "linear_p_ratio_mae": float(np.abs(group.true_p_ratio - group.pred_p_ratio).mean()),
            "valid": len(group), "total": len(group),
            "true_residual_sd_median": float(group.true_transverse_residual_sd.median()),
            "pred_residual_sd_median": float(group.pred_transverse_residual_sd.median()),
            "predicted_over_true_residual_sd_median": float(amplitude_ratio.median()),
            "residual_amplitude_pearson": correlation(group.true_transverse_residual_sd, group.pred_transverse_residual_sd),
            "residual_waveform_pearson_median": float(group.residual_waveform_pearson.median()),
            "true_residual_vs_log_temperature_pearson": correlation(group.true_transverse_residual_sd, np.log10(group.temperature_K)),
            "pred_residual_vs_log_temperature_pearson": correlation(group.pred_transverse_residual_sd, np.log10(group.temperature_K)),
        })
    pd.DataFrame(summary).to_csv(analysis / "summary.csv", index=False)

    cached = pd.read_csv(folder / "test/per_network.csv")
    final = table.merge(cached.query("frame==199")[["source", "trajectory_id", "mode", "true_p_ratio", "pred_p_ratio"]],
                        on=["source", "trajectory_id", "mode"], suffixes=("_linear", "_endpoint"), validate="one_to_one")
    true_error = float(np.abs(final.endpoint_true_p - final.true_p_ratio_endpoint).max())
    pred_error = float(np.abs(final.endpoint_pred_p - final.pred_p_ratio_endpoint).max())
    assert max(true_error, pred_error) < 2e-8
    after = {"ae": sha(SOURCE_AE), "propagator": sha(folder / "propagator.pt")}
    assert before == after
    (analysis / "completed.json").write_text(json.dumps({
        "status": "completed", "job_id": os.environ.get("PBS_JOBID"),
        "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
        "endpoint_parity_max_abs_true": true_error, "endpoint_parity_max_abs_pred": pred_error,
        "method": "normalized boundary-strain OLS frames30..horizon; residual diagnostics use transverse residual after identical fit",
        "interpretation": "p-ratio tests mean response; residual amplitude and waveform separately test thermal fluctuation reproduction",
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
