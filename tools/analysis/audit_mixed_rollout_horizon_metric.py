#!/usr/bin/env python3
"""Audit cumulative versus fixed-window mixed-T rollout p-ratio metrics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PREFIX_CODE = ROOT / "hpc/experiments/mini_temperature1000_linear_prefix"
ALL4_CODE = ROOT / "notebooks/results/all4_coverage_1000/code_v5"
for path in (PREFIX_CODE, ALL4_CODE, ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
spec = importlib.util.spec_from_file_location("frozen_prefix_metric_audit", PREFIX_CODE / "run.py")
frozen = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(frozen)

BASE = ROOT / "notebooks/results/all4_coverage_1000"
MODEL = BASE / "all4_twenty_each_retry_v5_validation30"
PREFIX = BASE / "prefix_linear_fit/all4_twenty_each_retry_v5_validation30/per_network_prefix.csv"
OUT = BASE / "rollout_metric_audit"
SOURCES = ("node_mixed", "stiffness_mixed")
WINDOW = 30
HORIZONS = tuple(range(59, 200))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    assert not OUT.exists(), OUT
    OUT.mkdir()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    before = {name: sha(MODEL / name) for name in ("ae.pt", "propagator.pt")}
    recipe = json.loads((MODEL / "propagator_recipe.json").read_text())
    result = frozen.load_experiment_bundle(MODEL / "propagator.pt", cfg=recipe["config"], device="cpu")
    rows = []
    with torch.no_grad():
        for source in SOURCES:
            simulations = frozen.load_sims(result, manifest["datasets"][source], source)
            for simulation in simulations:
                reference = simulation[0]
                physical_reference, _ = frozen._pref(reference)
                sides = frozen.directional_side_indices_from_box(physical_reference, quantile=.10)
                xy = physical_reference.x[:, :2].numpy()
                width0 = float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean())
                height0 = float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean())

                def widths(graph):
                    values = frozen.physical(graph, reference).x[:, :2].numpy()
                    return (float(values[sides["right"], 0].mean() - values[sides["left"], 0].mean()),
                            float(values[sides["top"], 1].mean() - values[sides["bottom"], 1].mean()))

                truth = [(width0, height0)]
                rollout = [(width0, height0)]
                z = frozen.encode_frame_latent(result["ae"], simulation, 0, pos_dim=2,
                    node_feature_mode=result["params"]["node_feature_mode"],
                    normalizers=result["normalizers"], device="cpu")
                context = frozen.encode_reference_context(result["ae"], simulation, pos_dim=2,
                    normalizers=result["normalizers"], device="cpu", include_temperature=False,
                    include_source_id=False, pool_mode="mean")
                for frame in range(1, 200):
                    truth.append(widths(simulation[frame]))
                    z = frozen.latent_step(result["dyn"], z, result["latent_stats"], context=context, loss_mode="delta")
                    graph = frozen.decode_latent_to_graph(result["ae"], simulation, z, frame, pos_dim=2,
                        ae_target_mode=result["params"]["ae_target_mode"],
                        normalizers=result["normalizers"], device="cpu")
                    rollout.append(widths(graph))
                tw, th = np.asarray(truth).T
                pw, ph = np.asarray(rollout).T
                tx, ty = (tw-width0)/width0, (th-height0)/height0
                px, py = (pw-width0)/width0, (ph-height0)/height0
                _, final_slope, final_fit_r2, final_pairs = frozen.ols(tx[30:200], ty[30:200])
                for horizon in HORIZONS:
                    start = horizon - WINDOW + 1
                    _, true_slope, true_fit_r2, true_pairs = frozen.ols(tx[start:horizon+1], ty[start:horizon+1])
                    _, pred_slope, pred_fit_r2, pred_pairs = frozen.ols(px[start:horizon+1], py[start:horizon+1])
                    rows.append({"source": source, "trajectory_id": int(reference.source_sim_id),
                        "horizon": horizon, "md_step": horizon*1000, "window_start": start,
                        "window_end": horizon, "window_pairs": WINDOW,
                        "true_window_p": -true_slope, "pred_window_p": -pred_slope,
                        "true_window_fit_r2": true_fit_r2, "pred_window_fit_r2": pred_fit_r2,
                        "true_window_pairs": true_pairs, "pred_window_pairs": pred_pairs,
                        "true_final_p": -final_slope, "true_final_fit_r2": final_fit_r2,
                        "true_final_pairs": final_pairs})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "fixed_window_per_network.csv", index=False)
    summaries = []
    cumulative = pd.read_csv(PREFIX).query("mode == 'autonomous_delta_mlp'")
    for (source, horizon), group in table.groupby(["source", "horizon"]):
        cumulative_group = cumulative.query("source == @source and horizon == @horizon")
        summaries.append({"source": source, "horizon": horizon, "md_step": horizon*1000,
            "fixed_same_horizon_r2": frozen.r2(group.true_window_p, group.pred_window_p),
            "fixed_truth_vs_final_r2": frozen.r2(group.true_final_p, group.true_window_p),
            "fixed_pred_vs_final_r2": frozen.r2(group.true_final_p, group.pred_window_p),
            "fixed_truth_vs_final_pearson": group.true_window_p.corr(group.true_final_p),
            "fixed_pred_vs_final_pearson": group.pred_window_p.corr(group.true_final_p),
            "fixed_true_fit_r2_median": float(group.true_window_fit_r2.median()),
            "fixed_pred_fit_r2_median": float(group.pred_window_fit_r2.median()),
            "cumulative_same_horizon_r2": frozen.r2(cumulative_group.linear_true_p, cumulative_group.linear_pred_p),
            "valid": len(group), "total": len(group)})
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    after = {name: sha(MODEL / name) for name in ("ae.pt", "propagator.pt")}
    assert before == after
    receipt = {"status": "completed", "job_id": os.environ.get("PBS_JOBID"),
        "checkpoint_sha256_before": before, "checkpoint_sha256_after": after,
        "sources": list(SOURCES), "heldout_ids": [40,119], "horizons": [59,199],
        "fixed_window_frames": WINDOW, "final_truth_fit_frames": [30,199],
        "postfit_only": True, "script_sha256": sha(Path(__file__))}
    (OUT / "completed.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
