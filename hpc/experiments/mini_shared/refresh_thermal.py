#!/usr/bin/env python3
"""Refresh only mixed-temperature post-fit metrics from frozen mini_shared models."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from collect import latents
from evaluate import evaluate
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/mini_shared"
THERMAL = OUT / "thermal_fit"

MAIN_ARMS = ("stiffness_only", "joint_original", "joint_all")
SINGLE_ARMS = ("single_stiffness", "single_node")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_sims(result: dict, entry: dict, name: str, cache: str, indices: list[int], offset: int):
    path_key, hash_key = f"{cache}_path", f"{cache}_sha256"
    if cache not in {"test", "trainval"}:
        raise ValueError(f"Unknown cache: {cache}")
    if sha(Path(entry[path_key])) != entry[hash_key]:
        raise ValueError(f"Input data hash changed for {name}/{cache}.")
    spec = {
        "name": name,
        "label": name,
        "path": entry[path_key],
        "train_count": 0,
        "val_count": len(indices),
        "split_indices": {"train": [], "val": indices, "test": []},
    }
    source = {
        "name": name,
        "label": name,
        "path": entry[path_key],
        "dataset_name": name,
        "source_name": name,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [spec],
    }
    _, sims, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
    if len(sims) != len(indices):
        raise ValueError(f"Unexpected {name} test count: {len(sims)}.")
    for index, sim in zip(indices, sims):
        for frame in sim:
            frame.source_sim_id = index + offset
    return sims


def copy_dir(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def verify_endpoint_identity(old_path: Path, refreshed: pd.DataFrame) -> dict:
    """Require the old endpoint predictions before replacing only mixed metrics."""
    old = pd.read_csv(old_path)
    keys = ["source", "trajectory_id", "frame", "mode"]
    expected = old[keys + ["true_p_ratio", "pred_p_ratio", "displacement_sse"]].rename(columns={
        "true_p_ratio": "true_p_ratio_old_endpoint",
        "pred_p_ratio": "pred_p_ratio_old_endpoint",
        "displacement_sse": "displacement_sse_old",
    })
    actual = refreshed[keys + ["true_p_ratio_endpoint", "pred_p_ratio_endpoint", "displacement_sse"]].rename(columns={
        "displacement_sse": "displacement_sse_new",
    })
    merged = expected.merge(actual, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Endpoint cohort mismatch for {old_path}: {merged['_merge'].value_counts().to_dict()}")
    result = {"old_path": str(old_path), "rows": int(len(merged)), "tolerance": {"rtol": 1e-10, "atol": 1e-10}}
    for label, old_column, new_column in (
        ("true_p_ratio", "true_p_ratio_old_endpoint", "true_p_ratio_endpoint"),
        ("pred_p_ratio", "pred_p_ratio_old_endpoint", "pred_p_ratio_endpoint"),
        ("displacement_sse", "displacement_sse_old", "displacement_sse_new"),
    ):
        equal = bool(np.allclose(merged[old_column].to_numpy(float), merged[new_column].to_numpy(float), rtol=1e-10, atol=1e-10, equal_nan=True))
        result[label + "_equal"] = equal
        if not equal:
            raise ValueError(f"Frozen endpoint mismatch for {old_path}: {label}")
    return result


def refresh_main(arm: str, result: dict, manifest: dict, output: Path) -> list[dict]:
    source = OUT / arm / "test"
    target = output / "test"
    target.mkdir()
    old_rows, old_steps, old_latents = [], [], []
    for name in ("stiffness_original", "node_original"):
        copy_dir(source / name, target / name)
        old_rows.append(pd.read_csv(source / name / "per_network.csv"))
        old_steps.append(pd.read_csv(source / name / "per_step.csv"))
    cached_latents = pd.read_csv(source / "latents.csv")
    old_latents.append(cached_latents.loc[cached_latents["source"].isin(("stiffness_original", "node_original"))].copy())
    old_latents[0].to_csv(target / "original_latents.csv", index=False)

    mixed_rows, mixed_steps, mixed_latents, checks = [], [], [], []
    for name in ("stiffness_mixed", "node_mixed"):
        sims = source_sims(result, manifest["datasets"][name], name, "test", list(range(80)), 40)
        rows = evaluate({**result, "val_data": sims}, seed=123, latent_dim=2, output=target / name, horizon=199, split="test")
        checks.append(verify_endpoint_identity(source / name / "per_network.csv", rows))
        mixed_rows.append(rows)
        mixed_steps.append(pd.read_csv(target / name / "per_step.csv"))
        value = latents(result, sims)
        value.to_csv(target / name / "latents.csv", index=False)
        mixed_latents.append(value)
        del sims
        gc.collect()
    pd.concat([*old_rows, *mixed_rows], ignore_index=True).to_csv(target / "per_network.csv", index=False)
    pd.concat([*old_steps, *mixed_steps], ignore_index=True).to_csv(target / "per_step.csv", index=False)
    pd.concat([*old_latents, *mixed_latents], ignore_index=True).to_csv(target / "latents.csv", index=False)
    return checks


def refresh_single(arm: str, result: dict, manifest: dict, output: Path) -> list[dict]:
    source = OUT / arm
    copy_dir(source / "demo", output / "demo")
    copy_dir(source / "unseen_transfer" / "original", output / "unseen_transfer" / "original")
    category = {"single_stiffness": "stiffness", "single_node": "node"}[arm]
    checks = []
    for name, cache, indices, offset, destination, label in (
        (f"{category}_mixed", "trainval", [0], 0, "paired_transfer", "paired_temperature_transfer"),
        (f"{category}_mixed", "test", list(range(20)), 40, "unseen_transfer/mixed", "unseen_mixed_temperature"),
    ):
        sims = source_sims(result, manifest["datasets"][name], name, cache, indices, offset)
        target = output / destination
        rows = evaluate({**result, "val_data": sims}, seed=123, latent_dim=2, output=target, horizon=199, split="test")
        checks.append(verify_endpoint_identity(source / destination / "per_network.csv", rows))
        rows["split"] = label
        rows.to_csv(target / "per_network.csv", index=False)
        value = latents(result, sims)
        value.to_csv(target / "latents.csv", index=False)
        if len(indices) == 1:
            (target / "per_step.csv").unlink()
        del sims
        gc.collect()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=(*MAIN_ARMS, *SINGLE_ARMS), required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    source = OUT / args.arm
    output = THERMAL / args.arm
    if output.exists():
        raise FileExistsError(output)
    if json.loads((source / "fit_completed.json").read_text())["status"] != "completed":
        raise ValueError(f"Frozen fit is incomplete: {args.arm}")
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Data manifest is incomplete.")
    recipe_path = source / "propagator_recipe.json"
    recipe = json.loads(recipe_path.read_text())
    ae_path, propagator_path = source / "ae.pt", source / "propagator.pt"
    before = {"ae_sha256": sha(ae_path), "propagator_sha256": sha(propagator_path)}
    output.mkdir(parents=True)
    dump(output / "refresh_recipe.json", {
        "arm": args.arm,
        "operation": "mixed-temperature post-fit metric refresh only; no model training or checkpoint selection",
        "frozen_source": str(source),
        "frozen_propagator_recipe_sha256": sha(recipe_path),
        "model_sha256_before": before,
        "input_data_sha256": {name: {"trainval": entry["trainval_sha256"], "test": entry["test_sha256"]} for name, entry in manifest["datasets"].items()},
        "test_used_for_selection": False,
    })
    result = load_experiment_bundle(propagator_path, cfg=recipe["config"], device="cpu")
    if args.arm in MAIN_ARMS:
        checks = refresh_main(args.arm, result, manifest, output)
    else:
        checks = refresh_single(args.arm, result, manifest, output)
    after = {"ae_sha256": sha(ae_path), "propagator_sha256": sha(propagator_path)}
    if before != after:
        raise RuntimeError("Frozen checkpoint bytes changed during evaluation.")
    dump(output / "endpoint_identity_check.json", {"status": "passed", "checks": checks})
    dump(output / "completed.json", {
        "status": "completed",
        "arm": args.arm,
        "model_sha256_before": before,
        "model_sha256_after": after,
        "input_data_sha256": {name: {"trainval": entry["trainval_sha256"], "test": entry["test_sha256"]} for name, entry in manifest["datasets"].items()},
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    main()
