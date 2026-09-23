#!/usr/bin/env python3
"""Evaluate frames 1, 5, 10, and 25 for frozen compact MP2D AEs.

This deliberately loads only the declared validation trajectories and frames
before applying the frozen snapshot's canonical preprocessing.  It never
loads the training pool, trains, or selects a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from graph_utils.box import Box
import graph_utils.pratio as metric_module
from lss.data import (POSITION_NORMALIZATION, append_lj_edge_indicator,
                      normalize_dataset_coordinates, normalize_dataset_edges,
                      tag_simulation_source)
from lss.graph import clone_graph
from lss.dynamics.experiment import _autoencoder_class
from lss.dynamics.training import decode_latent_to_graph, encode_frame_latent
from lss.dynamics.simulation import r2_score

SEEDS = (3456456, 123, 456, 786, 2026)
FRAMES = (1, 5, 10, 25)
SOURCES = ("reid", "depablo_low_temp", "lj_noisy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical(graph, reference):
    if str(getattr(reference, "coordinate_normalization", "")).lower() != POSITION_NORMALIZATION:
        raise ValueError("Expected position-normalized graph.")
    out = clone_graph(graph).cpu()
    half = reference.reference_box_half_extent.detach().cpu().double()[:2]
    center = reference.reference_box_center.detach().cpu().double()[:2]
    out.x = out.x.detach().cpu().double().clone()
    out.x[:, :2] = out.x[:, :2] * half + center
    box = out.box
    out.box = Box(float(box.x1 * half[0] + center[0]), float(box.x2 * half[0] + center[0]),
                  float(box.y1 * half[1] + center[1]), float(box.y2 * half[1] + center[1]),
                  float(getattr(box, "z1", -.1)), float(getattr(box, "z2", .1)))
    out.box_tensor = half * torch.as_tensor(graph.box_tensor, dtype=torch.float64)[:2]
    out.coordinate_normalization = "physical"
    return out


def load_model(checkpoint: Path):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    params = {**bundle["params"], **bundle["source_spec"]}
    stats = bundle["stats"]
    normalizers = {key: stats[key].cpu() for key in (
        "target_mean", "target_std", "node_feature_mean", "node_feature_std", "edge_mean", "edge_std")}
    for key in ("ref_edge_mean", "ref_edge_std"):
        if key in stats:
            normalizers[key] = stats[key].cpu()
    model = _autoencoder_class(str(params["autoencoder_model"]).lower())(
        pos_dim=int(params["pos_dim"]), node_feature_dim=int(normalizers["node_feature_mean"].numel()),
        edge_dim=int(normalizers["edge_mean"].numel()), hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]), latent_tokens=int(params["latent_tokens"]),
        reconstruction_dim=int(normalizers["target_mean"].numel()),
        message_passing_steps=int(params["message_passing_steps"])).cpu()
    model.set_edge_normalization(normalizers["edge_mean"], normalizers["edge_std"],
                                 normalizers["ref_edge_mean"], normalizers["ref_edge_std"])
    model.edge_mode = str(params["edge_mode"])
    incompatible = model.load_state_dict(bundle["ae_state_dict"], strict=False)
    if incompatible.unexpected_keys or (int(params["message_passing_steps"]) and incompatible.missing_keys):
        raise RuntimeError(f"Incompatible checkpoint: {incompatible}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return params, model, normalizers


def load_selected(spec: dict):
    """Select raw trajectories/frames before canonical edge and coordinate transforms."""
    ids = list(spec["split_indices"]["val"])
    raw = torch.load(Path(spec["path"]), map_location="cpu", weights_only=False)
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError(f"Invalid validation IDs for {spec['name']}")
    selected = [[trajectory[frame] for frame in (0, *FRAMES)] for trajectory in (raw[index] for index in ids)]
    normalize_dataset_edges(selected, edge_multiplicity=1, edge_vector_dim=2)
    if bool(spec.get("append_lj_indicator", False)) or spec.get("lj_max_graph_distance") is not None:
        append_lj_edge_indicator(selected, max_graph_distance=spec.get("lj_max_graph_distance"), pos_dim=2)
    normalize_dataset_coordinates(selected, mode="position_normalization", pos_dim=2)
    return [tag_simulation_source(sim, str(spec["name"])) for sim in selected], ids


def evaluate_seed(seed: int, cohorts: dict[str, tuple[list, list[int]]], all_rows: list[dict]):
    run = ROOT / f"notebooks/results/compact_lj_mp2d/mp2_r16_d2_s{seed}_code_v4"
    checkpoint, recipe_path = run / "ae.pt", run / "recipe.json"
    recipe, completed = json.loads(recipe_path.read_text()), json.loads((run / "completed.json").read_text())
    checkpoint_hash, recipe_hash = sha256(checkpoint), sha256(recipe_path)
    if completed["ae_file_sha256"] != checkpoint_hash or recipe["seed"] != seed:
        raise ValueError(f"Frozen provenance mismatch for seed {seed}")
    if recipe["response_selection"] or recipe["final_test_used"]:
        raise ValueError("Evaluation contract violation in training recipe")
    params, ae, normalizers = load_model(checkpoint)
    mapping = {frame: idx for idx, frame in enumerate((0, *FRAMES))}
    expected = {"reid", "depablo_low_temp", "lj_noisy"}
    specs = recipe["source"]["dataset_mixture"]
    if {item["name"] for item in specs} != expected:
        raise ValueError("Unexpected source cohort")
    for spec in specs:
        cached_ids = cohorts[spec["name"]][1]
        if list(spec["split_indices"]["val"]) != cached_ids:
            raise ValueError(f"Seed {seed} validation IDs differ for {spec['name']}")
    with torch.no_grad():
        for spec in specs:
            source = spec["name"]
            sims, ids = cohorts[source]
            for original_index, sim in zip(ids, sims):
                ref = sim[0]
                if source == "lj_noisy":
                    added = ref.edge_attr[:, 4] > .5
                    if not added.any() or not torch.all(ref.edge_attr[added, 3] == 0):
                        raise ValueError("LJ distance-three augmentation mismatch")
                pref = physical(ref, ref)
                inverse_error = float((pref.x[:, :2] - ref.reference_context_positions.cpu().double()[:, :2]).abs().max())
                if inverse_error > 2e-5:
                    raise ValueError(f"Reference inverse mismatch {inverse_error}")
                pref.x[:, :2] = ref.reference_context_positions.cpu().double()[:, :2]
                sides = directional_side_indices_from_box(pref, quantile=.10)
                for original_frame, compact_frame in mapping.items():
                    if original_frame == 0:
                        continue
                    z = encode_frame_latent(ae, sim, compact_frame, pos_dim=2,
                        node_feature_mode=params["node_feature_mode"], normalizers=normalizers, device="cpu")
                    pred = decode_latent_to_graph(ae, sim, z, compact_frame, pos_dim=2,
                        ae_target_mode=params["ae_target_mode"], normalizers=normalizers, device="cpu")
                    truth_phys, pred_phys = physical(sim[compact_frame], ref), physical(pred, ref)
                    true_p = float(calc_p_ratio_rollout_sides([pref, truth_phys], -1, side_idx=sides))
                    pred_p = float(calc_p_ratio_rollout_sides([pref, pred_phys], -1, side_idx=sides))
                    all_rows.append(dict(variant=recipe["variant"], strategy="ae", kind="reconstruction", seed=seed,
                        source=source, original_index=int(original_index), frame=original_frame,
                        true_p_ratio=true_p, pred_p_ratio=pred_p, finite_true=math.isfinite(true_p),
                        finite_pred=math.isfinite(pred_p), checkpoint_sha256=checkpoint_hash,
                        training_recipe_sha256=recipe_hash, inverse_reference_max_abs_error=inverse_error))
    if sha256(checkpoint) != checkpoint_hash or sha256(recipe_path) != recipe_hash:
        raise ValueError("Frozen input changed during evaluation")


def main():
    start = time.time(); torch.set_num_threads(1)
    out = ROOT / "notebooks/results/mp2_latent_analysis/early_response"
    if any((out / name).exists() for name in ("per_network_rows.csv", "response_seed_results.csv", "metadata.json", "completed.json")):
        raise FileExistsError(f"Refusing to overwrite completed results in {out}")
    first_recipe = json.loads((ROOT / "notebooks/results/compact_lj_mp2d/mp2_r16_d2_s123_code_v4/recipe.json").read_text())
    cohorts = {}
    for spec in first_recipe["source"]["dataset_mixture"]:
        cohorts[spec["name"]] = load_selected(spec)
    print(json.dumps({"stage": "loaded_selected_cohorts", "sources": {key: len(value[0]) for key, value in cohorts.items()}}), flush=True)
    rows = []
    for seed in SEEDS:
        print(json.dumps({"stage": "evaluating_seed", "seed": seed}), flush=True)
        evaluate_seed(seed, cohorts, rows)
    table = pd.DataFrame(rows)
    if len(table) != 5 * 3 * 20 * 4:
        raise ValueError(f"Unexpected row count {len(table)}")
    summaries = []
    for (seed, source, frame), group in table.groupby(["seed", "source", "frame"], sort=True):
        true = group.true_p_ratio.to_numpy(); pred = group.pred_p_ratio.to_numpy()
        summaries.append(dict(variant="mp2_r16_d2", strategy="ae", kind="reconstruction", seed=int(seed), source=source,
            frame=int(frame), p_ratio_r2=r2_score(true, pred), valid=int((np.isfinite(true) & np.isfinite(pred)).sum()),
            total=int(len(group)), true_nonfinite=int((~np.isfinite(true)).sum()), pred_nonfinite=int((~np.isfinite(pred)).sum())))
    summary = pd.DataFrame(summaries)
    if len(summary) != 60 or set(summary.frame) != set(FRAMES):
        raise ValueError("Summary coverage mismatch")
    # The retained frame 25 is a strict, per-network truth and prediction parity gate.
    parity_max = {}
    for seed in SEEDS:
        old = pd.read_csv(ROOT / f"notebooks/results/compact_lj_mp2d/pratio/mp2_r16_d2_s{seed}_code_v4/per_network_rows.csv")
        old = old[old.frame.eq(25) & old.source.isin(SOURCES)]
        new = table[(table.seed == seed) & (table.frame == 25)]
        paired = new.merge(old, on=["source", "original_index", "frame"], validate="one_to_one", suffixes=("_new", "_old"))
        if len(old) != 60 or len(paired) != 60:
            raise ValueError(f"Frame-25 parity coverage failure seed {seed}: {len(old)}/{len(paired)}")
        delta = max(float((paired.true_p_ratio_new - paired.true_p_ratio_old).abs().max()),
                    float((paired.pred_p_ratio_new - paired.pred_p_ratio_old).abs().max()))
        if not math.isfinite(delta) or delta > 1e-10:
            raise ValueError(f"Frame-25 truth/prediction parity failure seed {seed}: {delta}")
        parity_max[str(seed)] = delta
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_network_rows.csv", index=False)
    summary.to_csv(out / "response_seed_results.csv", index=False)
    metadata = dict(status="completed", frames=list(FRAMES), sources=list(SOURCES), seeds=list(SEEDS), rows=len(table),
        summary_rows=len(summary), checkpoint_training_recipe="frozen MP2D code_v4 ae.pt and recipe.json per seed",
        source_preprocessing="raw selected validation trajectories and frames 0/1/5/10/25; canonical snapshot edges, LJ augmentation, then reference-box normalization",
        frame_mapping={str(k): v for k, v in {frame: idx for idx, frame in enumerate((0, *FRAMES))}.items()},
        frame25_max_abs_truth_or_prediction_delta=parity_max, metric="physical endpoint sides quantile=0.10",
        metric_module_sha256=sha256(Path(metric_module.__file__)), runner_sha256=sha256(Path(__file__)),
        canonical_code=str(ROOT / "src/lss/dynamics"), python=sys.version, torch_version=torch.__version__, platform=platform.platform(),
        source_files={spec["name"]: {"path": spec["path"], "sha256": sha256(Path(spec["path"])), "validation_ids": spec["split_indices"]["val"]}
                      for spec in first_recipe["source"]["dataset_mixture"]},
        training_runs={str(seed): {"checkpoint": str(ROOT / f"notebooks/results/compact_lj_mp2d/mp2_r16_d2_s{seed}_code_v4/ae.pt"),
            "recipe": str(ROOT / f"notebooks/results/compact_lj_mp2d/mp2_r16_d2_s{seed}_code_v4/recipe.json"),
            "training_job_id": json.loads((ROOT / f"notebooks/results/compact_lj_mp2d/mp2_r16_d2_s{seed}_code_v4/completed.json").read_text())["job_id"]} for seed in SEEDS},
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        seconds=time.time()-start, training_performed=False, selection_performed=False, final_test_used=False,
        failures=[])
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "completed", "summary_rows": len(summary), "parity": parity_max}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failed_out = ROOT / "notebooks/results/mp2_latent_analysis/early_response"
        failed_out.mkdir(parents=True, exist_ok=True)
        (failed_out / "failed.json").write_text(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "runner_sha256": sha256(Path(__file__))}, indent=2) + "\n")
        raise
