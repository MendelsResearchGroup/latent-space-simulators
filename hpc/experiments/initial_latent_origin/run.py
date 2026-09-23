"""Post-fit audit: learned versus inherited frame-zero p-ratio ordering."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import build_autoencoder, resolve_train_val_test
from lss.dynamics.simulation import (
    batch_delta_graphs,
    compact_reference_edge_features,
    reference_positions_for_model,
)
from lss.dynamics.training import encode_frame_latent

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/all4_coverage_1000"
MODEL_DIR = BASE / "all4_one_each_retry_v5_validation30"
OUT = ROOT / "notebooks/results/initial_latent_origin_z"
SOURCES = ("node_low", "stiffness_low")
RANDOM_SEEDS = (0, 1, 2, 3, 4, 17, 23, 42, 123, 456, 786, 2026, 3456456, 918273, 314159)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson(x, y) -> float:
    return float(np.corrcoef(np.asarray(x, dtype=float), np.asarray(y, dtype=float))[0, 1])


def pc1_stats(z: np.ndarray, response: np.ndarray) -> dict[str, float]:
    centered = z - z.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    score = centered @ vt[0]
    variance = singular.square() if isinstance(singular, torch.Tensor) else singular**2
    effective = float(variance.sum() ** 2 / np.square(variance).sum())
    return {
        "pc1_abs_pearson": abs(pearson(score, response)),
        "effective_dimension": effective,
        "pc1_variance_fraction": float(variance[0] / variance.sum()),
    }


def load_source(result, manifest, name: str):
    entry = manifest["datasets"][name]
    spec = {
        "name": name,
        "label": name,
        "path": entry["test_path"],
        "train_count": 0,
        "val_count": 80,
        "split_indices": {"train": [], "val": list(range(40, 120)), "test": []},
    }
    source = {
        "name": name,
        "label": name,
        "path": entry["test_path"],
        "dataset_name": name,
        "source_name": name,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [spec],
    }
    _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    if len(simulations) != 80:
        raise AssertionError(f"{name}: expected 80 held-out simulations, got {len(simulations)}")
    return simulations


def descriptors(sim) -> dict[str, float]:
    graph = sim[0]
    position = reference_positions_for_model(graph, pos_dim=2, device="cpu").numpy().astype(float)
    edge = compact_reference_edge_features(graph, pos_dim=2, device="cpu").numpy().astype(float)
    edge_index = graph.edge_index.numpy()
    vector, length, stiffness = edge[:, :2], edge[:, 2], edge[:, 3]
    unit = vector / np.maximum(length[:, None], 1e-12)
    degree = np.bincount(edge_index.reshape(-1), minlength=len(position)).astype(float)
    centered = position - position.mean(0)
    covariance = centered.T @ centered / len(centered)
    trace = float(np.trace(covariance))
    nx, ny = unit.T
    total_k = max(float(stiffness.sum()), 1e-12)
    return {
        "node_count": float(len(position)),
        "edge_count": float(edge_index.shape[1]),
        "mean_degree": float(degree.mean()),
        "std_degree": float(degree.std()),
        "x_span": float(np.ptp(position[:, 0])),
        "y_span": float(np.ptp(position[:, 1])),
        "gyration_radius": float(np.sqrt(trace)),
        "position_anisotropy": float((covariance[0, 0] - covariance[1, 1]) / max(trace, 1e-12)),
        "position_xy": float(covariance[0, 1] / max(trace, 1e-12)),
        "mean_edge_length": float(length.mean()),
        "std_edge_length": float(length.std()),
        "mean_stiffness": float(stiffness.mean()),
        "std_stiffness": float(stiffness.std()),
        "edge_anisotropy": float(np.mean(nx**2 - ny**2)),
        "edge_xy": float(np.mean(nx * ny)),
        "stiffness_anisotropy": float(np.sum(stiffness * (nx**2 - ny**2)) / total_k),
        "stiffness_xy": float(np.sum(stiffness * nx * ny) / total_k),
    }


def random_model(result, seed: int):
    torch.manual_seed(seed)
    params, normalizers = result["params"], result["normalizers"]
    model = build_autoencoder(params, edge_dim=int(normalizers["edge_mean"].numel()), device="cpu")
    model.set_edge_normalization(
        normalizers["edge_mean"], normalizers["edge_std"],
        normalizers.get("ref_edge_mean"), normalizers.get("ref_edge_std"),
    )
    model.edge_mode = str(params["edge_mode"])
    model.eval()
    return model


def normalized_frame0_inputs(result, simulation):
    params, norm = result["params"], result["normalizers"]
    batch = batch_delta_graphs(
        [simulation], [(0, 0)], pos_dim=2, device="cpu",
        node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"],
    )
    ref_mean = norm.get("ref_edge_mean", norm["edge_mean"])
    ref_std = norm.get("ref_edge_std", norm["edge_std"])
    return {
        "node": (batch["node_feature"] - norm["node_feature_mean"]) / norm["node_feature_std"],
        "ref_pos": batch["ref_pos"],
        "edge": (batch["edge_attr"] - norm["edge_mean"]) / norm["edge_std"],
        "ref_edge": (batch["ref_edge_attr"] - ref_mean) / ref_std,
        "edge_index": batch["edge_index"], "batch": batch["batch"],
    }


def encode_inputs(model, inputs, *, ref_pos=None, ref_edge=None):
    z, _ = model.encode(
        inputs["node"], inputs["ref_pos"] if ref_pos is None else ref_pos,
        inputs["edge"], inputs["ref_edge"] if ref_edge is None else ref_edge,
        inputs["edge_index"], inputs["batch"],
    )
    return z[0]


def main() -> None:
    start = time.monotonic()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "completed.json").exists():
        raise FileExistsError(f"Completed audit already exists: {OUT / 'completed.json'}")
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    recipe = json.loads((MODEL_DIR / "propagator_recipe.json").read_text())
    result = load_experiment_bundle(MODEL_DIR / "propagator.pt", cfg=recipe["config"], device="cpu")
    checkpoint_before = sha(MODEL_DIR / "ae.pt")
    response_table = pd.read_csv(MODEL_DIR / "test/per_network.csv")
    response_table = response_table.query(
        "mode == 'ae_reconstruction' and frame == 199 and source in @SOURCES"
    )[["source", "trajectory_id", "true_p_ratio"]].rename(columns={"true_p_ratio": "final_p_ratio"})
    cached_latent = pd.read_csv(MODEL_DIR / "test/latents.csv").query(
        "frame == 0 and source in @SOURCES"
    )

    source_data = {name: load_source(result, manifest, name) for name in SOURCES}
    descriptor_rows = []
    for source, simulations in source_data.items():
        for trajectory_id, simulation in zip(range(40, 120), simulations):
            descriptor_rows.append({"source": source, "trajectory_id": trajectory_id, **descriptors(simulation)})
    descriptor_table = pd.DataFrame(descriptor_rows).merge(
        response_table, on=["source", "trajectory_id"], validate="one_to_one"
    )
    descriptor_table.to_csv(OUT / "static_descriptors.csv", index=False)

    descriptor_correlations = []
    descriptor_pc_rows = []
    descriptor_columns = [c for c in descriptor_table if c not in {"source", "trajectory_id", "final_p_ratio"}]
    for source, group in descriptor_table.groupby("source"):
        for column in descriptor_columns:
            values = group[column].to_numpy(float)
            descriptor_correlations.append({
                "source": source, "descriptor": column, "pearson": pearson(values, group.final_p_ratio),
                "abs_pearson": abs(pearson(values, group.final_p_ratio)), "valid": len(group), "total": len(group),
            })
        matrix = group[descriptor_columns].to_numpy(float)
        varying = matrix.std(0) > 1e-12
        standardized = (matrix[:, varying] - matrix[:, varying].mean(0)) / matrix[:, varying].std(0)
        descriptor_pc_rows.append({"source": source, **pc1_stats(standardized, group.final_p_ratio.to_numpy()), "descriptor_count": int(varying.sum())})
    pd.DataFrame(descriptor_correlations).to_csv(OUT / "descriptor_correlations.csv", index=False)
    pd.DataFrame(descriptor_pc_rows).to_csv(OUT / "descriptor_pc1.csv", index=False)

    trained_rows = []
    attribution_rows = []
    attribution_directions = {}
    for source, group in cached_latent.merge(response_table, on=["source", "trajectory_id"]).groupby("source"):
        z_values = group[["z0", "z1"]].to_numpy()
        trained_rows.append({"source": source, "model": "trained", "seed": 123,
                             "z0_abs_pearson": abs(pearson(z_values[:, 0], group.final_p_ratio)),
                             "z1_abs_pearson": abs(pearson(z_values[:, 1], group.final_p_ratio)),
                             **pc1_stats(z_values, group.final_p_ratio.to_numpy())})
        _, _, vt = np.linalg.svd(z_values - z_values.mean(0), full_matrices=False)
        attribution_directions[source] = vt[0]

    trained_model = result["ae"]
    for parameter in trained_model.parameters():
        parameter.requires_grad_(False)
    for source, simulations in source_data.items():
        direction = torch.tensor(attribution_directions[source], dtype=torch.float32)
        responses = response_table.query("source == @source").sort_values("trajectory_id").final_p_ratio.to_numpy()
        for trajectory_id, simulation, response in zip(range(40, 120), simulations, responses):
            inputs = normalized_frame0_inputs(result, simulation)
            with torch.no_grad():
                baseline = encode_inputs(trained_model, inputs)
                ref_edge_geometry_mean = inputs["ref_edge"].clone(); ref_edge_geometry_mean[:, :3] = 0
                ref_edge_stiffness_mean = inputs["ref_edge"].clone(); ref_edge_stiffness_mean[:, 3] = 0
                all_ref_edge_mean = torch.zeros_like(inputs["ref_edge"])
                centered_ref_pos = inputs["ref_pos"] - inputs["ref_pos"].mean(0, keepdim=True)
                zero_ref_pos = torch.zeros_like(inputs["ref_pos"])
                variants = {
                    "baseline": baseline,
                    "edge_geometry_at_fitted_mean": encode_inputs(trained_model, inputs, ref_edge=ref_edge_geometry_mean),
                    "stiffness_at_fitted_mean": encode_inputs(trained_model, inputs, ref_edge=ref_edge_stiffness_mean),
                    "all_reference_edges_at_fitted_mean": encode_inputs(trained_model, inputs, ref_edge=all_ref_edge_mean),
                    "reference_position_centered": encode_inputs(trained_model, inputs, ref_pos=centered_ref_pos),
                    "reference_position_zero_ood": encode_inputs(trained_model, inputs, ref_pos=zero_ref_pos),
                }
            ref_pos = inputs["ref_pos"].detach().requires_grad_(True)
            ref_edge = inputs["ref_edge"].detach().requires_grad_(True)
            z_grad = encode_inputs(trained_model, inputs, ref_pos=ref_pos, ref_edge=ref_edge)
            grad_pos, grad_edge = torch.autograd.grad((z_grad * direction).sum(), (ref_pos, ref_edge))
            centered = ref_pos.detach() - ref_pos.detach().mean(0, keepdim=True)
            base_projection = float((baseline * direction).sum())
            common = {
                "source": source, "trajectory_id": trajectory_id, "final_p_ratio": float(response),
                "position_gradient_rms": float(grad_pos.square().mean().sqrt()),
                "position_gradient_times_centered_input_rms": float((grad_pos * centered).square().mean().sqrt()),
                "edge_vector_gradient_rms": float(grad_edge[:, :2].square().mean().sqrt()),
                "edge_length_gradient_rms": float(grad_edge[:, 2].square().mean().sqrt()),
                "stiffness_gradient_rms": float(grad_edge[:, 3].square().mean().sqrt()),
            }
            for variant, value in variants.items():
                attribution_rows.append({
                    **common, "variant": variant,
                    "z0": float(value[0]), "z1": float(value[1]),
                    "pc1_projection": float((value * direction).sum()),
                    "pc1_projection_change": float((value * direction).sum()) - base_projection,
                    "latent_shift": float(torch.linalg.vector_norm(value - baseline)),
                })
    attribution = pd.DataFrame(attribution_rows)
    attribution.to_csv(OUT / "trained_input_attribution.csv", index=False)
    attribution_summary = []
    for (source, variant), group in attribution.groupby(["source", "variant"]):
        attribution_summary.append({
            "source": source, "variant": variant, "networks": len(group),
            "z0_abs_pearson_with_p": abs(pearson(group.z0, group.final_p_ratio)),
            "z1_abs_pearson_with_p": abs(pearson(group.z1, group.final_p_ratio)),
            "pc1_abs_pearson_with_p": abs(pearson(group.pc1_projection, group.final_p_ratio)),
            "pc1_projection_change_rms": float(np.sqrt(np.mean(np.square(group.pc1_projection_change)))),
            "latent_shift_mean": float(group.latent_shift.mean()),
            "position_gradient_rms_mean": float(group.position_gradient_rms.mean()),
            "position_gradient_times_centered_input_rms_mean": float(group.position_gradient_times_centered_input_rms.mean()),
            "edge_vector_gradient_rms_mean": float(group.edge_vector_gradient_rms.mean()),
            "edge_length_gradient_rms_mean": float(group.edge_length_gradient_rms.mean()),
            "stiffness_gradient_rms_mean": float(group.stiffness_gradient_rms.mean()),
        })
    pd.DataFrame(attribution_summary).to_csv(OUT / "trained_input_attribution_summary.csv", index=False)

    random_rows = []
    for seed in RANDOM_SEEDS:
        model = random_model(result, seed)
        for source, simulations in source_data.items():
            z = []
            with torch.no_grad():
                for simulation in simulations:
                    z.append(encode_frame_latent(
                        model, simulation, 0, pos_dim=2,
                        node_feature_mode=result["params"]["node_feature_mode"],
                        normalizers=result["normalizers"], device="cpu",
                    ).numpy())
            response = response_table.query("source == @source").sort_values("trajectory_id").final_p_ratio.to_numpy()
            z_array=np.stack(z)
            random_rows.append({"source": source, "model": "random", "seed": seed,
                                "z0_abs_pearson": abs(pearson(z_array[:, 0], response)),
                                "z1_abs_pearson": abs(pearson(z_array[:, 1], response)),
                                **pc1_stats(z_array, response)})
        print("random seed", seed, "complete", flush=True)
    controls = pd.concat([pd.DataFrame(trained_rows), pd.DataFrame(random_rows)], ignore_index=True)
    controls.to_csv(OUT / "encoder_controls.csv", index=False)

    summary_rows = []
    for source in SOURCES:
        trained = controls.query("source == @source and model == 'trained'").iloc[0]
        random = controls.query("source == @source and model == 'random'")
        summary_rows.append({
            "source": source,
            "trained_pc1_abs_pearson": trained.pc1_abs_pearson,
            "trained_effective_dimension": trained.effective_dimension,
            "random_seeds": len(random),
            "random_pc1_abs_pearson_mean": random.pc1_abs_pearson.mean(),
            "random_pc1_abs_pearson_sd": random.pc1_abs_pearson.std(ddof=1),
            "random_pc1_abs_pearson_p95": random.pc1_abs_pearson.quantile(.95),
            "random_pc1_abs_pearson_max": random.pc1_abs_pearson.max(),
            "random_effective_dimension_mean": random.effective_dimension.mean(),
            "trained_z0_abs_pearson": trained.z0_abs_pearson,
            "trained_z1_abs_pearson": trained.z1_abs_pearson,
            "random_z0_abs_pearson_mean": random.z0_abs_pearson.mean(),
            "random_z1_abs_pearson_mean": random.z1_abs_pearson.mean(),
            "random_z0_abs_pearson_max": random.z0_abs_pearson.max(),
            "random_z1_abs_pearson_max": random.z1_abs_pearson.max(),
            "trained_exceeds_random_max": bool(trained.pc1_abs_pearson > random.pc1_abs_pearson.max()),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "encoder_control_summary.csv", index=False)
    checkpoint_after = sha(MODEL_DIR / "ae.pt")
    if checkpoint_before != checkpoint_after:
        raise AssertionError("Frozen AE checkpoint changed during post-fit audit")
    metadata = {
        "status": "completed", "job_id": os.environ.get("PBS_JOBID"),
        "sources": list(SOURCES), "heldout_ids": list(range(40, 120)),
        "response": "cached physical endpoint p-ratio at frame199; post-fit diagnostic only",
        "random_seeds": list(RANDOM_SEEDS),
        "random_control": "same architecture and fitted input normalizers; fresh untrained weights; response-blind PC1; absolute sign-invariant Pearson reported",
        "descriptors": descriptor_columns,
        "descriptor_pc1": "per-source z-scored static descriptors; PCA uses no response labels",
        "trained_input_attribution": "frame0 trained-encoder PC1; normalized reference-edge block mean ablations, position centering/zero controls, and local input gradients; PC1 direction uses latent coordinates only",
        "attribution_qualification": "mean/zero interventions can be out of distribution and correlated optimized inputs prevent unique physical-causality assignment",
        "checkpoint_sha256_before": checkpoint_before, "checkpoint_sha256_after": checkpoint_after,
        "model_recipe_sha256": sha(MODEL_DIR / "recipe.json"),
        "data_manifest_sha256": sha(BASE / "data_manifest.json"),
        "script_sha256": sha(Path(__file__)), "postfit_only": True,
        "p_ratio_not_used_for_model_or_pc_direction": True,
        "seconds": time.monotonic() - start,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (OUT / "completed.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
