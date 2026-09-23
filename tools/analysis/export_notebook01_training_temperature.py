"""Post-fit training markers and mixed-temperature fluctuation audit for notebook 01."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_utils import directional_side_indices_from_box
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.training import encode_frame_latent


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "notebooks/results/all4_coverage_1000/all4_one_each_retry_v5_validation30"
OUT = ROOT / "notebooks/results/all4_coverage_1000/notebook01_training_temperature"
LATENTS = MODEL / "test/latents.csv"
MANIFESTS = {
    "node_mixed": ROOT.parent / "MetaForge/results/mini_temperature_1000/mixed/mini_node_optimized_200_500/manifest.json",
    "stiffness_mixed": ROOT.parent / "MetaForge/results/mini_temperature_1000/mixed/mini_stiff_optimized_200_1500/manifest.json",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_module():
    path = ROOT / "notebooks/results/all4_coverage_1000/code_v5/evaluate.py"
    spec = importlib.util.spec_from_file_location("notebook01_frozen_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def width_height(graph, sides) -> tuple[float, float]:
    xy = graph.x[:, :2].detach().cpu().numpy()
    return float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean()), float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean())


def linear_p(simulation, evaluate) -> tuple[float, float]:
    reference = simulation[0]
    physical_reference, _ = evaluate._pref(reference)
    sides = directional_side_indices_from_box(physical_reference, quantile=0.10)
    width0, height0 = width_height(physical_reference, sides)
    values = []
    for frame in range(200):
        graph = physical_reference if frame == 0 else evaluate.physical(simulation[frame], reference)
        width, height = width_height(graph, sides)
        values.append(((width - width0) / width0, (height - height0) / height0))
    x, y = np.asarray(values)[30:].T
    matrix = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
    fitted = matrix @ beta
    fit_r2 = 1.0 - np.square(y - fitted).sum() / np.square(y - y.mean()).sum()
    return float(-beta[1]), float(fit_r2)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    OUT.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((MODEL / "propagator_recipe.json").read_text())
    result = load_experiment_bundle(MODEL / "propagator.pt", cfg=recipe["config"], device="cpu")
    evaluate = evaluation_module()
    training = []
    for simulation in result["train_data"]:
        source = str(simulation[0].source_name)
        z = encode_frame_latent(result["ae"], simulation, 0, pos_dim=2, node_feature_mode=result["params"]["node_feature_mode"], normalizers=result["normalizers"], device="cpu").detach().cpu().reshape(-1)
        reference, _ = evaluate._pref(simulation[0])
        sides = directional_side_indices_from_box(reference, quantile=0.10)
        if source.endswith("_mixed"):
            response, fit_r2 = linear_p(simulation, evaluate)
            estimator = "normalized boundary-strain OLS frames 30-199"
        else:
            response = evaluate._p_ratio(reference, evaluate.physical(simulation[199], simulation[0]), sides)
            fit_r2 = float("nan")
            estimator = "physical endpoint frame 199"
        trajectory_id = int(getattr(simulation[0], "source_sim_id", getattr(simulation[0], "source_index")))
        training.append({"source": source, "trajectory_id": trajectory_id, "temperature_K": float(simulation[0].temperature), "initial_z0": float(z[0]), "initial_z1": float(z[1]), "training_p_ratio": response, "estimator": estimator, "within_trajectory_fit_r2": fit_r2, "positive_p_ratio": bool(response > 0)})
    training = pd.DataFrame(training).sort_values("source")
    assert len(training) == 4 and set(training.trajectory_id) == {0}
    training.to_csv(OUT / "training_trajectory_markers.csv", index=False)

    frames = pd.read_csv(LATENTS)
    fluctuation_rows = []
    for source, manifest_path in MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text())
        temperatures = {int(row["source_index"]): float(row["temperature_K"]) for row in manifest["records"]}
        selected = frames[(frames.source == source) & frames.frame.between(30, 199)]
        for trajectory_id, group in selected.groupby("trajectory_id"):
            x, y = group[["x_deformation", "y_deformation"]].to_numpy(float).T
            matrix = np.c_[np.ones(len(x)), x]
            residual = y - matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0]
            fluctuation_rows.append({"source": source, "trajectory_id": int(trajectory_id), "temperature_K": temperatures[int(trajectory_id)], "transverse_fluctuation_sd": float(np.std(residual, ddof=1)), "frames": len(group)})
    fluctuation = pd.DataFrame(fluctuation_rows)
    fluctuation.to_csv(OUT / "mixed_temperature_fluctuation.csv", index=False)
    summary = []
    for source, group in fluctuation.groupby("source"):
        summary.append({"source": source, "pearson_fluctuation_vs_log10_temperature": group.transverse_fluctuation_sd.corr(np.log10(group.temperature_K)), "spearman_fluctuation_vs_temperature": group.transverse_fluctuation_sd.corr(group.temperature_K, method="spearman"), "valid": len(group), "total": 80, "training_mixed_temperature_K": 0.1, "heldout_temperature_min_K": float(group.temperature_K.min()), "heldout_temperature_max_K": float(group.temperature_K.max())})
    pd.DataFrame(summary).to_csv(OUT / "mixed_temperature_fluctuation_summary.csv", index=False)
    metadata = {"status": "complete", "model": str(MODEL.relative_to(ROOT)), "checkpoint_sha256": sha(MODEL / "ae.pt"), "training_marker_method": "source-appropriate position-derived response; low endpoint frame199, mixed normalized-strain OLS frames30-199", "fluctuation_method": "sample SD of residual transverse displacement after affine OLS against axial displacement over frames30-199", "qualification": "Temperature association confirms that the heldout data contain increasing thermal fluctuations. Together with training mixed ID0 at 0.1K it documents out-of-training-temperature evaluation up to30K, but it does not alone prove causal temperature extrapolation because structure and temperature vary together across heldout IDs.", "input_sha256": {"latents": sha(LATENTS), **{source: sha(path) for source, path in MANIFESTS.items()}}, "script_sha256": sha(Path(__file__))}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(training.to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
