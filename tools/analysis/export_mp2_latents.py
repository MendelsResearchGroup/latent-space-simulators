#!/usr/bin/env python3
"""Export frozen MP2D validation-frame latents and physical response labels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from graph_utils.box import Box

from lss.dynamics.experiment import _load_ae_cache, resolve_train_val_test
from lss.dynamics.simulation import batch_delta_graphs
from lss.dynamics.training import ae_target_tensor
from lss.dynamics.simulation import canonical_edge_mode
from lss.data import POSITION_NORMALIZATION
from lss.graph import clone_graph


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "notebooks/results/compact_lj_mp2d/mp2_r16_d2_s786_code_v4"
OUT = ROOT / "notebooks/results/mp2_latent_analysis"
CHECKPOINT = RUN / "ae.pt"
RECIPE = RUN / "recipe.json"
COMPLETED = RUN / "completed.json"
FRAMES = tuple(range(200))
BATCH_SIZE = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical(graph, reference):
    """Invert the run's position normalization, preserving the physical box."""
    if str(getattr(reference, "coordinate_normalization", "")).lower() != POSITION_NORMALIZATION:
        raise ValueError("Expected a position-normalized reference graph.")
    out = clone_graph(graph).cpu()
    half = reference.reference_box_half_extent.detach().cpu().double()[:2]
    center = reference.reference_box_center.detach().cpu().double()[:2]
    out.x = out.x.detach().cpu().double().clone()
    out.x[:, :2] = out.x[:, :2] * half + center
    box = out.box
    out.box = Box(
        float(box.x1 * half[0] + center[0]), float(box.x2 * half[0] + center[0]),
        float(box.y1 * half[1] + center[1]), float(box.y2 * half[1] + center[1]),
        float(getattr(box, "z1", -0.1)), float(getattr(box, "z2", 0.1)),
    )
    out.box_tensor = half * torch.as_tensor(graph.box_tensor, dtype=torch.float64)[:2]
    out.coordinate_normalization = "physical"
    return out


def extents(graph, sides) -> np.ndarray:
    pos = graph.x[:, :2].detach().cpu().numpy()
    return np.array([
        pos[sides["right"], 0].mean() - pos[sides["left"], 0].mean(),
        pos[sides["top"], 1].mean() - pos[sides["bottom"], 1].mean(),
    ])


def encode_batch(ae, sims, rows, *, params, normalizers, device) -> np.ndarray:
    """Canonical batched equivalent of ``encode_frame_latent``."""
    packed = batch_delta_graphs(
        sims, rows, pos_dim=2, device=device,
        node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"],
    )
    node = (packed["node_feature"] - normalizers["node_feature_mean"].to(device)) / normalizers["node_feature_std"].to(device)
    edge = (packed["edge_attr"] - normalizers["edge_mean"].to(device)) / normalizers["edge_std"].to(device)
    ref_edge = (packed["ref_edge_attr"] - normalizers["ref_edge_mean"].to(device)) / normalizers["ref_edge_std"].to(device)
    z, _ = ae.encode(node, packed["ref_pos"], edge, ref_edge, packed["edge_index"], packed["batch"])
    if z.shape != (len(rows), 2):
        raise ValueError(f"Unexpected batched latent shape {tuple(z.shape)}.")
    return z.detach().cpu().numpy()


def cohort(result: dict, recipe: dict) -> list[tuple[str, list, list[int]]]:
    groups = []
    for spec in recipe["source"]["dataset_mixture"]:
        source = spec["name"]
        sims = [sim for sim in result["val_data"] if str(sim[0].source_name) == source]
        ids = list(spec["split_indices"]["val"])
        if len(sims) != len(ids) or len(ids) != 20:
            raise ValueError(f"Validation cohort mismatch for {source}: {len(sims)}/{len(ids)}.")
        groups.append((source, sims, ids))
    mixed = recipe["mixed_evaluation"]
    params = {**result["params"], **mixed}
    train, val, test, _ = resolve_train_val_test(mixed, params, split_seed=params["split_seed"])
    ids = list(mixed["dataset_mixture"][0]["split_indices"]["val"])
    if train or test or len(val) != len(ids) or len(ids) != 20:
        raise ValueError("Mixed-T must be its exact post-fit 20-trajectory validation cohort.")
    groups.append(("depablo_mixed_temp", val, ids))
    return groups


def network_descriptors(ref) -> dict[str, float | int]:
    edge_attr = ref.edge_attr.detach().cpu()
    if edge_attr.ndim != 2 or edge_attr.size(1) != 5:
        raise ValueError(f"Expected compact five-channel edges, got {tuple(edge_attr.shape)}.")
    spring = edge_attr[:, 4] < 0.5
    stiffness = edge_attr[spring, 3]
    if stiffness.numel() == 0:
        raise ValueError("Network has no physical spring edges.")
    node_count = int(ref.x.size(0))
    spring_count = int(spring.sum())
    return {
        "mean_stiffness": float(stiffness.mean()),
        "mean_degree": float(2.0 * spring_count / node_count),
        "node_count": node_count,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing export directory: {OUT}")
    for path in (CHECKPOINT, RECIPE, COMPLETED):
        if not path.is_file():
            raise FileNotFoundError(path)
    recipe = json.loads(RECIPE.read_text())
    completed = json.loads(COMPLETED.read_text())
    if recipe["seed"] != 786 or recipe["config"]["ae_config"]["latent_dim"] != 2:
        raise ValueError("This exporter is restricted to frozen MP2D seed 786.")
    if recipe["response_selection"] or recipe["final_test_used"]:
        raise ValueError("Frozen-run provenance contract failed.")
    if completed["ae_file_sha256"] != sha256(CHECKPOINT):
        raise ValueError("Checkpoint hash does not match completed run record.")

    device = torch.device("cpu")
    result = _load_ae_cache(CHECKPOINT, dict(recipe["config"]), device=device)
    ae, normalizers, params = result["ae"], result["normalizers"], result["params"]
    if canonical_edge_mode(params["edge_mode"]) != "compact_stored":
        raise ValueError(f"Expected compact_stored preprocessing, got {params['edge_mode']}.")
    groups = cohort(result, recipe)
    if {source for source, _, _ in groups} != {"reid", "depablo_low_temp", "lj_noisy", "depablo_mixed_temp"}:
        raise ValueError("Unexpected source cohort.")

    frame_rows: list[dict] = []
    network_rows: list[dict] = []
    with torch.no_grad():
        for source, sims, simulation_ids in groups:
            if source == "lj_noisy" and not all((sim[0].edge_attr[:, 4] > 0.5).any() for sim in sims):
                raise ValueError("LJ graph-distance augmentation missing from validation data.")
            rows = [(sim_idx, frame) for sim_idx in range(len(sims)) for frame in FRAMES]
            latent = np.concatenate([
                encode_batch(ae, sims, rows[start:start + BATCH_SIZE], params=params, normalizers=normalizers, device=device)
                for start in range(0, len(rows), BATCH_SIZE)
            ])
            if latent.shape != (20 * 200, 2) or not np.isfinite(latent).all():
                raise ValueError(f"Invalid latent export for {source}: {latent.shape}.")
            for sim_idx, (simulation_id, sim) in enumerate(zip(simulation_ids, sims)):
                ref = sim[0]
                if len(sim) != 200:
                    raise ValueError(f"Expected 200 frames, got {len(sim)} for {source}/{simulation_id}.")
                pref = physical(ref, ref)
                inverse_error = float((pref.x[:, :2] - ref.reference_context_positions.cpu().double()[:, :2]).abs().max())
                if inverse_error > 2e-5:
                    raise ValueError(f"Physical inversion mismatch {inverse_error} for {source}/{simulation_id}.")
                pref.x[:, :2] = ref.reference_context_positions.cpu().double()[:, :2]
                sides = directional_side_indices_from_box(pref, quantile=0.10)
                initial_extent = extents(pref, sides)
                z = latent[sim_idx * 200:(sim_idx + 1) * 200]
                descriptors = network_descriptors(ref)
                p_final = float("nan")
                for frame in FRAMES:
                    truth = physical(sim[frame], ref)
                    strain = (extents(truth, sides) - initial_extent) / initial_extent
                    p_ratio = float("nan") if frame == 0 else float(calc_p_ratio_rollout_sides([pref, truth], -1, side_idx=sides))
                    if frame == 199:
                        p_final = p_ratio
                    frame_rows.append({
                        "source": source, "simulation_id": int(simulation_id), "frame": frame,
                        "z0": float(z[frame, 0]), "z1": float(z[frame, 1]),
                        "p_ratio": p_ratio, "strain_x": float(strain[0]), "strain_y": float(strain[1]),
                    })
                network_rows.append({
                    "source": source, "simulation_id": int(simulation_id),
                    "z0_initial": float(z[0, 0]), "z1_initial": float(z[0, 1]),
                    "p_ratio_final": p_final, **descriptors,
                })

    frame_table = pd.DataFrame(frame_rows)
    network_table = pd.DataFrame(network_rows)
    if len(frame_table) != 16000 or len(network_table) != 80:
        raise ValueError(f"Coverage mismatch: frames={len(frame_table)}, networks={len(network_table)}.")
    counts = frame_table.groupby("source", sort=True).size().to_dict()
    if set(counts.values()) != {4000} or (frame_table.frame == 0).sum() != 80 or frame_table.loc[frame_table.frame == 0, "p_ratio"].notna().any():
        raise ValueError("Frame/source coverage or undefined frame-zero p-ratio contract failed.")

    OUT.mkdir(parents=True)
    frame_table.to_csv(OUT / "frame_rows.csv", index=False)
    network_table.to_csv(OUT / "network_rows.csv", index=False)
    metadata = {
        "variant": "mp2_r16_d2", "model_seed": 786, "split_seed": recipe["config"]["split_seed"],
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "recipe_sha256": sha256(RECIPE), "normalizers_sha256": sha256(RUN / "normalizers.json"),
        "dataset_sha256": recipe["dataset_sha256"], "manifest_sha256": recipe["manifest_sha256"],
        "frame_selection": "all frames 0 through 199 inclusive", "frame_rows": len(frame_table),
        "network_rows": len(network_table), "per_source_frame_rows": counts,
        "per_source_network_rows": network_table.groupby("source", sort=True).size().to_dict(),
        "p_ratio_estimator": "physical_endpoint_sides_quantile_0.10; frame 0 retained with NaN",
        "strain_estimator": "physical directional-side extents versus frame-zero physical reference, quantile=0.10",
        "coordinate_preprocessing": "saved split normalizers; canonical compact_stored edges; LJ graph-distance-three augmentation",
        "response_selection": False, "final_test_used": False,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
