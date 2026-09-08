"""Full-horizon compact-reference AE reconstruction runner.

This script intentionally trains no propagator.  Response quantities are not
read as features or targets for training, checkpoint selection, or the reconstruction
controls recorded here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
CODE = Path(os.environ["LSS_CODE_ROOT"])
VERSION = os.environ["LSS_CODE_VERSION"]
RESULTS_ROOT = Path(os.environ.get("LSS_RESULTS_ROOT", ROOT / "notebooks/results"))

def load_portable_manifest(path: Path) -> dict:
    """Keep historical evidence untouched while locating copied datasets."""
    manifest = json.loads(path.read_text())
    for spec in manifest.get("sources", {}).values():
        spec["path"] = str(ROOT / "data" / Path(spec["path"]).name)
    return manifest
sys.path.insert(0, str(CODE / "src"))

import numpy as np
import pandas as pd
import torch

from lss.data import append_lj_edge_indicator, load_dataset
from lss.latent.experiment import (
    resolve_train_val_test,
    run_latent_experiment,
    seed_everything,
)
from lss.latent.training import decode_latent_to_graph, encode_frame_latent


SEEDS = (3456456, 123, 456, 786, 2026)
VARIANTS = {
    "r16_d2": ("attention_reference16_corrected", 2, True),
    "r16_d4": ("attention_reference16_corrected", 4, True),
    "r16_d6": ("attention_reference16_corrected", 6, True),
    "r16_d8": ("attention_reference16_corrected", 8, True),
    "r96_d4": ("orientation_corrected", 4, True),
    "r96_d8": ("orientation_corrected", 8, True),
    "mp2_r16_d4": ("message_passing_reference16", 4, True),
    "mp2_r16_d8": ("message_passing_reference16", 8, True),
    "ljonly_r16_d4": ("attention_reference16_corrected", 4, False),
    "ljonly_r16_d8": ("attention_reference16_corrected", 8, False),
}
FULL_FRAMES = (0, 5, 10, 25, 50, 75, 100, 125, 149, 175, 199)
SMOKE_FRAMES = (0, 1, 3)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))


def dataset_spec(manifest: dict, name: str, *, train_count: int, val_count: int) -> dict:
    source = manifest["sources"][name]
    return {
        "name": name,
        "label": name,
        "path": source["path"],
        "train_count": int(train_count),
        "val_count": int(val_count),
        "split_indices": {
            "train": list(source["split_indices"]["train"][:train_count]),
            "val": list(source["split_indices"]["val"][:val_count]),
            "test": [],
        },
        # Five columns are used for every source.  The last column identifies
        # added LJ relations, not a response or physics-derived target.
        "append_lj_indicator": True,
        "lj_max_graph_distance": 3 if name == "lj_noisy" else None,
    }


def build_recipe(args: argparse.Namespace, manifest: dict, output: Path) -> tuple[dict, dict, dict]:
    model, latent_dim, shared = VARIANTS[args.variant]
    retained_train = 2 if args.smoke else 30
    retained_val = 2 if args.smoke else 20
    lj_train = 2 if args.smoke else 60
    lj_val = 2 if args.smoke else 20
    names = ("reid", "depablo_low_temp", "lj_noisy") if shared else ("lj_noisy",)
    specs = [
        dataset_spec(
            manifest,
            name,
            train_count=lj_train if name == "lj_noisy" else retained_train,
            val_count=lj_val if name == "lj_noisy" else retained_val,
        )
        for name in names
    ]
    transfer_count = 2 if args.smoke else 20
    mixed_spec = dataset_spec(
        manifest, "depablo_mixed_temp", train_count=0, val_count=transfer_count
    ) if shared else None
    source = {
        "dataset_name": "compact_lj_shared" if shared else "compact_lj_only",
        "source_name": "reid_lowT_lj" if shared else "lj_only",
        "label": "shared Reid + low-T + noisy-LJ" if shared else "noisy-LJ only",
        "path": specs[0]["path"],
        "dataset_mixture": specs,
        "coordinate_normalization": "position_normalization",
    }
    frames = 4 if args.smoke else 200
    ae_config = {
        "model": model,
        "latent_dim": latent_dim,
        "latent_tokens": 32,
        "hidden_size": 96,
        "edge_feature_dim": 5,
        "node_feature_mode": "normalized_delta",
        "target_mode": "normalized_delta",
        "max_train_frames_per_sim": frames,
        "max_val_frames_per_sim": frames,
        "max_epochs": 1 if args.smoke else 60,
        "patience": 1 if args.smoke else 12,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "mix_sources": True,
        "balance_sources": False,
        "gradient_method": "source_mean",
        "checkpoint_metric": "val_max_source_reconstruction",
        "checkpoint_mode": "min",
        "pratio_eval_every": 0,
    }
    if model == "message_passing_reference16":
        ae_config["message_passing_steps"] = 2
    cfg = {
        "ae_config": ae_config,
        "coordinate_normalization": "position_normalization",
        "edge_mode": "compact_stored",
        "static_context_use_physical_reference": True,
        "pos_dim": 2,
        "batch_graphs": 32,
        "frame_skip": 1,
        "early_stop_min_delta": 1e-5,
        "split_seed": 123,
        "model_seed": args.seed,
        "should_train_propagator": False,
        "should_rollout": False,
        "force_train": True,
        "cache_require_matching_config": True,
        "cache_path": str(output / "ae.pt"),
    }
    mixed = {
        "dataset_name": "compact_lj_mixed_transfer",
        "source_name": "depablo_mixed_temp",
        "label": "mixed-T post-fit transfer only",
        "path": mixed_spec["path"],
        "dataset_mixture": [mixed_spec],
        "coordinate_normalization": "position_normalization",
    } if shared else None
    return source, cfg, mixed


def _edge_key(index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    lo, hi = torch.minimum(index[0], index[1]), torch.maximum(index[0], index[1])
    return lo * n_nodes + hi


def verify_lj_edge_schema(spec: dict, *, required_frames: int) -> dict:
    """Check raw relation augmentation before normalizers or model reloads."""
    raw = load_dataset(spec["path"], coordinate_normalization=None, pos_dim=2)
    train_id = int(spec["split_indices"]["train"][0])
    if len(raw) < spec["train_count"] + spec["val_count"] or train_id >= len(raw):
        raise ValueError("LJ source has fewer trajectories than its declared split budget.")
    selected = raw[train_id]
    trajectory_length = len(selected)
    if trajectory_length < required_frames:
        raise ValueError("LJ source is shorter than the requested training horizon.")
    original = copy.deepcopy(selected[0])
    del raw, selected
    gc.collect()
    checked = copy.deepcopy(original)
    holder = [[checked]]
    append_lj_edge_indicator(holder, max_graph_distance=3, pos_dim=2)
    if checked.edge_attr.ndim != 2 or checked.edge_attr.size(1) != 5:
        raise ValueError("Compact LJ schema must have five raw edge feature columns.")
    if not bool(getattr(checked, "lj_edge_indicator_appended", False)):
        raise ValueError("LJ relation indicator was not added.")
    original_keys = _edge_key(original.edge_index, original.x.size(0))
    augmented_keys = _edge_key(checked.edge_index, checked.x.size(0))
    positions_by_key = {int(key): position for position, key in enumerate(augmented_keys.tolist())}
    if any(int(key) not in positions_by_key for key in original_keys.tolist()):
        raise ValueError("Original LJ spring edges were not preserved during augmentation.")
    original_positions = torch.tensor(
        [positions_by_key[int(key)] for key in original_keys.tolist()], dtype=torch.long
    )
    original_rows = checked.edge_attr[original_positions]
    if not torch.allclose(original_rows[:, :4], original.edge_attr[:, :4]):
        raise ValueError("Original LJ edge attributes changed during augmentation.")
    if not torch.equal(original_rows[:, 4], torch.zeros_like(original_rows[:, 4])):
        raise ValueError("Original spring edges must retain LJ indicator zero.")
    added = checked.edge_attr[:, 4] > 0.5
    if not added.any() or not torch.allclose(checked.edge_attr[added, 3], torch.zeros_like(checked.edge_attr[added, 3])):
        raise ValueError("Added LJ relations must have indicator one and raw stiffness zero.")
    index_once, attr_once = checked.edge_index.clone(), checked.edge_attr.clone()
    append_lj_edge_indicator(holder, max_graph_distance=3, pos_dim=2)
    if not torch.equal(index_once, checked.edge_index) or not torch.equal(attr_once, checked.edge_attr):
        raise ValueError("Repeated augmentation of the same graph was not idempotent.")
    return {
        "declared_train_id": train_id,
        "trajectory_length": trajectory_length,
        "raw_edge_count": int(original.edge_index.size(1)),
        "augmented_edge_count": int(checked.edge_index.size(1)),
        "added_edge_count": int(added.sum()),
        "edge_feature_schema": str(getattr(checked, "edge_feature_schema", "")),
        "all_added_stiffness_zero": True,
        "all_original_indicator_zero": True,
    }


def verify_schema(sims: list, *, source: str, required_frames: int) -> None:
    for sim in sims:
        if len(sim) < required_frames:
            raise ValueError(f"{source} trajectory shorter than {required_frames} frames.")
        reference = sim[0]
        if reference.edge_attr.ndim != 2 or reference.edge_attr.size(1) != 5:
            raise ValueError(f"{source} does not have the required five-channel edge schema.")
        if source != "lj_noisy" and not torch.equal(
            reference.edge_attr[:, 4], torch.zeros_like(reference.edge_attr[:, 4])
        ):
            raise ValueError(f"{source} must have zero LJ-relation indicators.")


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.detach().cpu() - target.detach().cpu()).square().mean())


def diagnostic_rows(result: dict, groups: list[tuple[str, str, list, list[int]]], frames: tuple[int, ...]) -> pd.DataFrame:
    ae, params, normalizers = result["ae"], result["params"], result["normalizers"]
    ae.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for split, source, sims, identities in groups:
            if not sims:
                continue
            for ordinal, sim in enumerate(sims):
                for frame in frames:
                    if frame >= len(sim):
                        raise ValueError(f"Diagnostic frame {frame} is unavailable for {source}.")
                    z = encode_frame_latent(
                        ae, sim, frame, pos_dim=2,
                        node_feature_mode=params["node_feature_mode"],
                        normalizers=normalizers, device="cpu",
                    )
                    pred = decode_latent_to_graph(
                        ae, sim, z, frame, pos_dim=2,
                        ae_target_mode=params["ae_target_mode"], normalizers=normalizers,
                        device="cpu",
                    )
                    shuffled = sims[(ordinal + 1) % len(sims)]
                    z_shuffled = encode_frame_latent(
                        ae, shuffled, frame, pos_dim=2,
                        node_feature_mode=params["node_feature_mode"],
                        normalizers=normalizers, device="cpu",
                    )
                    shuffled_pred = decode_latent_to_graph(
                        ae, sim, z_shuffled, frame, pos_dim=2,
                        ae_target_mode=params["ae_target_mode"], normalizers=normalizers,
                        device="cpu",
                    )
                    target = sim[frame].x[:, :2]
                    zero = sim[0].x[:, :2]
                    coordinate_mse = mse(pred.x[:, :2], target)
                    zero_mse = mse(zero, target)
                    rows.append({
                        "split": split,
                        "source": source,
                        "original_index": int(identities[ordinal]),
                        "frame": int(frame),
                        "coordinate_mse": coordinate_mse,
                        "zero_displacement_mse": zero_mse,
                        "explained_displacement_score": (
                            1.0 - coordinate_mse / zero_mse if zero_mse > 0 else np.nan
                        ),
                        "shuffled_same_source_coordinate_mse": mse(shuffled_pred.x[:, :2], target),
                        "shuffled_original_index": int(identities[(ordinal + 1) % len(sims)]),
                        "valid": int(np.isfinite(coordinate_mse)),
                        "total": 1,
                        "control": "same_source_cyclic_next_latent",
                        "transfer_only": split == "mixed_transfer",
                    })
    return pd.DataFrame(rows)


def aggregate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["split", "source", "frame", "transfer_only"], dropna=False).agg(
        networks=("original_index", "size"),
        valid=("valid", "sum"), total=("total", "sum"),
        coordinate_mse=("coordinate_mse", "mean"),
        zero_displacement_mse=("zero_displacement_mse", "mean"),
        explained_displacement_score=("explained_displacement_score", "mean"),
        shuffled_same_source_coordinate_mse=("shuffled_same_source_coordinate_mse", "mean"),
    ).reset_index()


def normalizer_json(normalizers: dict) -> dict:
    """Persist fitted scale parameters needed to assess LJ-only confounds."""
    return {
        name: value.detach().cpu().reshape(-1).tolist()
        for name, value in normalizers.items()
        if name in {
            "node_feature_mean", "node_feature_std", "target_mean", "target_std",
            "edge_mean", "edge_std", "ref_edge_mean", "ref_edge_std",
        }
    }


def exposure_summary(result: dict, cfg: dict) -> dict:
    """Record actual source sizes and transparent batch-step approximations."""
    frame_cap = int(cfg["ae_config"]["max_train_frames_per_sim"])
    batch_graphs = int(cfg["batch_graphs"])
    by_source: dict[str, dict] = {}
    for source in sorted({sim[0].source_name for sim in result["train_data"]}):
        sims = [sim for sim in result["train_data"] if sim[0].source_name == source]
        frames = sum(min(len(sim), frame_cap) for sim in sims)
        by_source[source] = {
            "trajectories": len(sims),
            "available_frames_min": min(len(sim) for sim in sims),
            "configured_frames_per_trajectory": frame_cap,
            "frame_samples_used": frames,
        }
    total = sum(item["frame_samples_used"] for item in by_source.values())
    history = result["ae_history"]
    return {
        "train_by_source": by_source,
        "total_train_frame_samples": total,
        "batch_graphs": batch_graphs,
        "approx_optimizer_steps_per_epoch": int(np.ceil(total / batch_graphs)),
        "epochs_completed": int(len(history)),
        "history_columns": list(history.columns),
    }


def bundle_reload_check(source: dict, cfg: dict, result: dict) -> dict:
    """Smoke-only cache reload check, including the reference-bottleneck models."""
    cfg_reload = copy.deepcopy(cfg)
    cfg_reload["force_train"] = False
    reloaded = run_latent_experiment(source, cfg_reload, device="cpu")
    sim = result["val_data"][0]
    frame = 1
    kwargs = dict(
        pos_dim=2, node_feature_mode=result["params"]["node_feature_mode"],
        normalizers=result["normalizers"], device="cpu",
    )
    before = encode_frame_latent(result["ae"], sim, frame, **kwargs)
    reload_kwargs = dict(
        pos_dim=2, node_feature_mode=reloaded["params"]["node_feature_mode"],
        normalizers=reloaded["normalizers"], device="cpu",
    )
    after = encode_frame_latent(reloaded["ae"], sim, frame, **reload_kwargs)
    decode_kwargs = dict(
        pos_dim=2, ae_target_mode=result["params"]["ae_target_mode"],
        normalizers=result["normalizers"], device="cpu",
    )
    before_graph = decode_latent_to_graph(result["ae"], sim, before, frame, **decode_kwargs)
    reload_decode_kwargs = dict(
        pos_dim=2, ae_target_mode=reloaded["params"]["ae_target_mode"],
        normalizers=reloaded["normalizers"], device="cpu",
    )
    after_graph = decode_latent_to_graph(reloaded["ae"], sim, after, frame, **reload_decode_kwargs)
    if not torch.allclose(before, after, atol=1e-6, rtol=1e-5):
        raise ValueError("Bundle reload changed encoded latent values.")
    if not torch.allclose(before_graph.x[:, :2], after_graph.x[:, :2], atol=1e-6, rtol=1e-5):
        raise ValueError("Bundle reload changed decoded coordinates.")
    return {"latent_equal": True, "decode_equal": True, "frame": frame}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    name = f"{args.variant}_s{args.seed}_{VERSION}" + ("_smoke" if args.smoke else "")
    output = RESULTS_ROOT / "compact_lj_reconstruction" / name
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        manifest_path = Path(os.environ.get("LSS_SPLIT_MANIFEST", RESULTS_ROOT / "lj_ae_08_bridge/split_manifest.json"))
        manifest = load_portable_manifest(manifest_path)
        source, cfg, mixed_source = build_recipe(args, manifest, output)
        required_frames = 4 if args.smoke else 200
        lj_spec = next(spec for spec in source["dataset_mixture"] if spec["name"] == "lj_noisy")
        recipe = {
            "source": source, "config": cfg, "mixed_evaluation": mixed_source,
            "variant": args.variant, "seed": args.seed, "smoke": args.smoke,
            "code_version": VERSION, "code_root": str(CODE),
            "job_id": os.environ.get("PBS_JOBID", "local"),
            "manifest_sha256": file_sha256(manifest_path),
            "dataset_sha256": {name: manifest["sources"][name]["sha256"] for name in manifest["sources"]},
            "code_hashes_sha256": file_sha256(CODE / "hashes.json") if (CODE / "hashes.json").exists() else None,
            "runner_sha256": file_sha256(Path(__file__)),
            "selection": "worst retained-source validation coordinate reconstruction only",
            "response_selection": False, "final_test_used": False,
            "shared_fit": args.variant not in {"ljonly_r16_d4", "ljonly_r16_d8"},
            "lj_only_caveat": "LJ-only differs in normalization and update budget from shared fitting; it is a diagnostic comparison, not causal matched proof.",
            "raw_lj_edge_control": "pending_preflight",
        }
        dump_json(output / "recipe.json", recipe)
        dump_json(output / "status.json", {"status": "running", "stage": "preflight"})
        specs_to_verify = [*source["dataset_mixture"], *([] if mixed_source is None else mixed_source["dataset_mixture"])]
        for spec in specs_to_verify:
            if file_sha256(Path(spec["path"])) != manifest["sources"][spec["name"]]["sha256"]:
                raise ValueError(f"Dataset changed: {spec['name']}")
        raw_lj_control = verify_lj_edge_schema(lj_spec, required_frames=required_frames)
        recipe["raw_lj_edge_control"] = raw_lj_control
        dump_json(output / "recipe.json", recipe)
        seed_everything(args.seed)
        result = run_latent_experiment(source, cfg, device="cpu")
        if result["test_data"]:
            raise ValueError("Final-test data must not be loaded.")
        if result["normalizers"]["edge_mean"].numel() != 5 or result["normalizers"].get("ref_edge_mean", result["normalizers"]["edge_mean"]).numel() != 5:
            raise ValueError("Model/reference context normalizers do not retain the five-channel edge schema.")
        if int(getattr(result["ae"], "edge_dim", 5)) != 5:
            raise ValueError("AE was not built with five-channel edge contexts.")
        if any(sim[0].source_name == "depablo_mixed_temp" for key in ("train_data", "val_data") for sim in result[key]):
            raise ValueError("Mixed-T entered fitting or retained-source validation.")
        for split_name in ("train_data", "val_data"):
            grouped: dict[str, list] = {}
            for sim in result[split_name]:
                grouped.setdefault(sim[0].source_name, []).append(sim)
            for source_name, sims in grouped.items():
                verify_schema(sims, source=source_name, required_frames=required_frames)
        result["ae_history"].to_csv(output / "ae_history.csv", index=False)
        dump_json(output / "normalizers.json", normalizer_json(result["normalizers"]))
        exposure = exposure_summary(result, cfg)
        dump_json(output / "training_exposure.json", exposure)
        mixed = []
        if mixed_source is not None:
            _, mixed, test, _ = resolve_train_val_test(mixed_source, result["params"], split_seed=cfg["split_seed"])
            if test or len(mixed) != (2 if args.smoke else 20):
                raise ValueError("Mixed-T transfer split does not match its declared post-fit scope.")
            verify_schema(mixed, source="depablo_mixed_temp", required_frames=required_frames)
        groups: list[tuple[str, str, list, list[int]]] = []
        for split, data, key, limit in (
            ("train", result["train_data"], "train", 2 if args.smoke else 5),
            ("validation", result["val_data"], "val", 2 if args.smoke else 20),
        ):
            for source_name in [spec["name"] for spec in source["dataset_mixture"]]:
                sims = [sim for sim in data if sim[0].source_name == source_name][:limit]
                ids = next(spec for spec in source["dataset_mixture"] if spec["name"] == source_name)["split_indices"][key][:len(sims)]
                if len(sims) != len(ids) or not sims:
                    raise ValueError(f"Missing {split} diagnostics for {source_name}.")
                groups.append((split, source_name, sims, ids))
        if mixed_source is not None:
            mixed_ids = mixed_source["dataset_mixture"][0]["split_indices"]["val"]
            groups.append(("mixed_transfer", "depablo_mixed_temp", mixed, mixed_ids))
        rows = diagnostic_rows(result, groups, SMOKE_FRAMES if args.smoke else FULL_FRAMES)
        rows.to_csv(output / "source_frame_rows.csv", index=False)
        aggregate_rows(rows).to_csv(output / "source_frame_summary.csv", index=False)
        reload_check = bundle_reload_check(source, cfg, result) if args.smoke else None
        checkpoint_sha256 = file_sha256(Path(cfg["cache_path"]))
        completed = {
            "status": "completed", "seconds": time.time() - started,
            "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "variant": args.variant, "seed": args.seed, "smoke": args.smoke,
            "parameters": sum(parameter.numel() for parameter in result["ae"].parameters()),
            "job_id": os.environ.get("PBS_JOBID", "local"),
            "output_horizon": 3 if args.smoke else 199,
            "epochs_completed": exposure["epochs_completed"],
            "total_train_frame_samples": exposure["total_train_frame_samples"],
            "approx_optimizer_steps_per_epoch": exposure["approx_optimizer_steps_per_epoch"],
            "ae_file_sha256": checkpoint_sha256,
            "bundle_reload": reload_check,
        }
        dump_json(output / "completed.json", completed)
        dump_json(output / "status.json", {"status": "completed", "stage": "complete", **completed})
    except Exception:
        failed = {
            "status": "failed", "seconds": time.time() - started,
            "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "variant": args.variant, "seed": args.seed, "smoke": args.smoke,
            "traceback": traceback.format_exc(),
        }
        dump_json(output / "failed.json", failed)
        dump_json(output / "status.json", {"status": "failed", "stage": "failed", **failed})
        raise


if __name__ == "__main__":
    main()
