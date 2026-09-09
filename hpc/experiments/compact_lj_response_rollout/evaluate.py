"""Physical-coordinate reconstruction and autonomous rollout evaluation."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from graph_utils.box import Box
from lss.data import POSITION_NORMALIZATION
from lss.graph import clone_graph
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import (
    decode_latent_to_graph, encode_frame_latent, encode_reference_context,
    latent_step,
)

FRAMES = (25, 50, 100, 150, 199)


def physical(graph, reference):
    """Invert position normalization using the original frame-zero reference."""
    if str(getattr(reference, "coordinate_normalization", "")).lower() != POSITION_NORMALIZATION:
        raise ValueError("Expected position-normalized reference graph.")
    out = clone_graph(graph).cpu()
    half = reference.reference_box_half_extent.detach().cpu().double()[:2]
    center = reference.reference_box_center.detach().cpu().double()[:2]
    out.x = out.x.detach().cpu().double().clone()
    out.x[:, :2] = out.x[:, :2] * half + center
    box = out.box
    out.box = Box(float(box.x1 * half[0] + center[0]), float(box.x2 * half[0] + center[0]),
                  float(box.y1 * half[1] + center[1]), float(box.y2 * half[1] + center[1]),
                  float(getattr(box, "z1", -0.1)), float(getattr(box, "z2", 0.1)))
    out.box_tensor = half * torch.as_tensor(graph.box_tensor, dtype=torch.float64)[:2]
    out.coordinate_normalization = "physical"
    return out


def _extents(graph, sides):
    pos = graph.x[:, :2].detach().cpu().numpy()
    return np.array([pos[sides["right"], 0].mean() - pos[sides["left"], 0].mean(),
                     pos[sides["top"], 1].mean() - pos[sides["bottom"], 1].mean()])


def mixed_validation(result: dict, mixed_spec: dict) -> list:
    """Load the forbidden-from-fit mixed-T cohort only after dynamics fit."""
    params = {**result["params"], **copy.deepcopy(mixed_spec)}
    train, val, test, _ = resolve_train_val_test(
        mixed_spec, params, split_seed=params["split_seed"]
    )
    if train or test or len(val) != 20:
        raise ValueError("Mixed-T must be a 20-trajectory post-fit validation-only cohort.")
    return val


def _row(*, kind, variant, strategy, seed, source, split, original_index, frame,
         truth, pred, reference, physical_reference, sides, initial_extent, checkpoint_sha256):
    truth_phys, pred_phys = physical(truth, reference), physical(pred, reference)
    true_p = float(calc_p_ratio_rollout_sides([physical_reference, truth_phys], -1, side_idx=sides))
    pred_p = float(calc_p_ratio_rollout_sides([physical_reference, pred_phys], -1, side_idx=sides))
    true_strain = (_extents(truth_phys, sides) - initial_extent) / initial_extent
    pred_strain = (_extents(pred_phys, sides) - initial_extent) / initial_extent
    return dict(variant=variant, strategy=strategy, seed=int(seed), source=source, split=split,
                original_index=int(original_index), frame=int(frame), kind=kind,
                true_p_ratio=true_p, pred_p_ratio=pred_p,
                coordinate_mse=float((pred.x[:, :2] - truth.x[:, :2]).square().mean()),
                finite_true=math.isfinite(true_p), finite_pred=math.isfinite(pred_p),
                finite_pair=math.isfinite(true_p) and math.isfinite(pred_p),
                true_strain_x=float(true_strain[0]), true_strain_y=float(true_strain[1]),
                pred_strain_x=float(pred_strain[0]), pred_strain_y=float(pred_strain[1]),
                checkpoint_sha256=checkpoint_sha256,
                p_ratio_method="physical_endpoint_sides_quantile_0.10")


def evaluate(result: dict, source_recipe: dict, *, variant: str, strategy: str,
             seed: int, checkpoint_sha256: str, device: str = "cpu") -> pd.DataFrame:
    """Initial-frame-only autonomous rollout plus frozen-AE reconstruction."""
    ae, dyn, stats, normalizers, params = (result[k] for k in ("ae", "dyn", "latent_stats", "normalizers", "params"))
    ae.eval(); dyn.eval()
    groups: list[tuple[str, str, list, list[int]]] = []
    for spec in source_recipe["source"]["dataset_mixture"]:
        sims = [s for s in result["val_data"] if str(s[0].source_name) == spec["name"]]
        ids = spec["split_indices"]["val"]
        if len(sims) != len(ids) or len(ids) != 20:
            raise ValueError(f"Validation cohort mismatch for {spec['name']}.")
        groups.append(("validation", spec["name"], sims, ids))
    mixed = source_recipe["mixed_evaluation"]
    groups.append(("mixed_transfer", "depablo_mixed_temp", mixed_validation(result, mixed),
                   mixed["dataset_mixture"][0]["split_indices"]["val"]))
    rows = []
    with torch.no_grad():
        for split, source, sims, ids in groups:
            for original_index, sim in zip(ids, sims):
                ref = sim[0]
                if len(sim) < 200 or ref.edge_attr.size(1) != 5:
                    raise ValueError("Expected full 200-frame compact-five-edge data.")
                if source == "lj_noisy":
                    if not (ref.edge_attr[:, 4] > 0.5).any():
                        raise ValueError("LJ graph-distance augmentation missing.")
                elif not torch.equal(ref.edge_attr[:, 4], torch.zeros_like(ref.edge_attr[:, 4])):
                    raise ValueError("Non-LJ source has an LJ relation indicator.")
                pref = physical(ref, ref)
                inverse_error = float((pref.x[:, :2] - ref.reference_context_positions.cpu().double()[:, :2]).abs().max())
                if inverse_error > 2e-5:
                    raise ValueError(f"Physical inverse error {inverse_error}.")
                pref.x[:, :2] = ref.reference_context_positions.cpu().double()[:, :2]
                sides = directional_side_indices_from_box(pref, quantile=.10)
                initial_extent = _extents(pref, sides)
                # Reconstruction independently encodes every target frame.
                for frame in FRAMES:
                    z = encode_frame_latent(ae, sim, frame, pos_dim=2,
                        node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device=device)
                    pred = decode_latent_to_graph(ae, sim, z, frame, pos_dim=2,
                        ae_target_mode=params["ae_target_mode"], normalizers=normalizers, device=device)
                    rows.append(_row(kind="reconstruction", variant=variant, strategy=strategy, seed=seed,
                        source=source, split=split, original_index=original_index, frame=frame,
                        truth=sim[frame], pred=pred, reference=ref, physical_reference=pref, sides=sides,
                        initial_extent=initial_extent, checkpoint_sha256=checkpoint_sha256))
                # One initial encoded state; all later states are model outputs.
                z = encode_frame_latent(ae, sim, 0, pos_dim=2,
                    node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device=device)
                context = encode_reference_context(ae, sim, pos_dim=2, normalizers=normalizers,
                    device=device, include_temperature=False, include_source_id=False, pool_mode="mean")
                for frame in range(1, 200):
                    z = latent_step(dyn, z, stats, context=context, loss_mode="delta")
                    if frame in FRAMES:
                        pred = decode_latent_to_graph(ae, sim, z, frame, pos_dim=2,
                            ae_target_mode=params["ae_target_mode"], normalizers=normalizers, device=device)
                        rows.append(_row(kind="autonomous", variant=variant, strategy=strategy, seed=seed,
                            source=source, split=split, original_index=original_index, frame=frame,
                            truth=sim[frame], pred=pred, reference=ref, physical_reference=pref, sides=sides,
                            initial_extent=initial_extent, checkpoint_sha256=checkpoint_sha256))
    return pd.DataFrame(rows)


def assert_d4_reconstruction_parity(rows: pd.DataFrame, *, seed: int,
                                    p_ratio_atol: float = 1e-5,
                                    coordinate_mse_atol: float = 1e-10) -> dict:
    """Require the new physical helper to reproduce the frozen D4 backfill."""
    root = Path(os.environ.get("LSS_PROJECT_ROOT", Path.cwd())).resolve()
    old_path = root / "notebooks/results/compact_lj_pratio/runs" / (
        f"compact_lj_reconstruction__mp2_r16_d4__s{seed}/per_network_rows.csv"
    )
    old = pd.read_csv(old_path)
    new = rows.loc[rows["kind"] == "reconstruction"].copy()
    keys = ["source", "split", "original_index", "frame"]
    merged = new.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(merged) != len(new) or len(new) != 400:
        raise ValueError(f"D4 reconstruction parity coverage mismatch {len(merged)}/{len(new)}.")
    deltas = {key: float((merged[f"{key}_new"] - merged[f"{key}_old"]).abs().max())
              for key in ("true_p_ratio", "pred_p_ratio", "coordinate_mse")}
    limits = {"true_p_ratio": p_ratio_atol, "pred_p_ratio": p_ratio_atol,
              "coordinate_mse": coordinate_mse_atol}
    if any(not math.isfinite(value) or value > limits[key] for key, value in deltas.items()):
        raise ValueError(f"D4 reconstruction parity mismatch: {deltas}")
    return {"rows": len(merged), **deltas}
