#!/usr/bin/env python3
"""Frozen AE+propagator endpoint-design study; physics is verification only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

DONOR = Path("/rg/mendels_prj/alexander.z/DL-course-project")
ROLL_CODE = DONOR / "notebooks/results/reference_simplification/code_v2"
ENDPOINT_CODE = DONOR / "notebooks/results/network_design/code_endpoint_v1"
ENDPOINT_HELPER = DONOR / "notebooks/network_design/endpoint_physics.py"
PREP = DONOR / "notebooks/results/network_design/calibrated_direction_v1"
sys.path[:0] = [
    str(DONOR.parent / "MetaForge/src"),
    str(ROLL_CODE / "src"),
    str(ENDPOINT_CODE / "notebooks/network_design"),
]

import auxetic  # noqa: F401  # install legacy pickle aliases before torch.load
import numpy as np
import pandas as pd
import torch
from graph_utils import directional_side_indices_from_box
from design import physical_candidate
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.training import (
    decode_latent_positions,
    encode_frame_latent,
    encode_reference_context,
    latent_step,
)

SOURCE = "depablo_low_temp"
NETWORKS = (70, 78, 96)
SEEDS = (786, 123)
BOUND = 2.0
STEPS = 150
LR = 0.02
EPS = 1e-12


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def stiffness_graph(reference, stiffness: torch.Tensor):
    """Replace model and physical-reference stiffness without detaching edits."""
    graph = reference.clone()
    if reference.edge_attr.ndim != 2 or reference.edge_attr.size(1) != 4:
        raise ValueError("Expected the frozen four-feature edge schema.")
    graph.edge_attr = torch.cat(
        [reference.edge_attr[:, :3], stiffness.reshape(-1, 1)], dim=1
    )
    context = getattr(reference, "reference_context_edge_attr", None)
    if not isinstance(context, torch.Tensor) or context.shape != reference.edge_attr.shape:
        raise ValueError("Missing compatible physical reference-edge metadata.")
    graph.reference_context_edge_attr = torch.cat(
        [context[:, :3], stiffness.reshape(-1, 1)], dim=1
    )
    return graph


def project_centered_box_(u: torch.Tensor, active: torch.Tensor, bound: float) -> None:
    """Project active edits onto sum(u)=0 and [-bound, bound]."""
    values = u[active].clone()
    lower, upper = float(values.min()) - bound, float(values.max()) + bound
    for _ in range(60):
        shift = (lower + upper) / 2.0
        if float((values - shift).clamp(-bound, bound).mean()) > 0:
            lower = shift
        else:
            upper = shift
    u[active] = (values - (lower + upper) / 2.0).clamp(-bound, bound)
    u[~active] = 0
    if abs(float(u[active].mean())) > 2e-6:
        raise RuntimeError("Zero-sum box projection failed.")


def frozen_bundle(seed: int):
    path = (
        DONOR
        / "notebooks/results/reference_simplification"
        / f"r16_d4_s{seed}_code_v2/context16/bundle.pt"
    )
    # Only restore model weights and saved normalization statistics. The
    # capacity loader's optional split cache avoids loading any trajectories.
    class ModelOnlySplitCache(dict):
        def get(self, key, default=None):
            return ([], [], [], [])
    result = load_experiment_bundle(path, cfg={}, device="cpu", split_cache=ModelOnlySplitCache())
    assert not result['train_data'] and not result['val_data'] and not result['test_data']
    ae, dyn = result["ae"].eval(), result["dyn"].eval()
    for model in (ae, dyn):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    params = result["params"]
    required = {
        "autoencoder_model": "attention_reference16",
        "latent_dim": 4,
        "node_feature_mode": "normalized_delta",
        "ae_target_mode": "normalized_delta",
        "edge_mode": "compact_stored",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_use_static_context": True,
        "graph_context_dim": 16,
    }
    if any(params.get(key) != value for key, value in required.items()):
        raise ValueError(f"Unexpected frozen rollout recipe for seed {seed}.")
    return path, result


def fixed_sides(physical_reference):
    """Freeze the original physical quantile-0.10 groups before optimization."""
    selected = directional_side_indices_from_box(physical_reference, quantile=0.10)
    return {key: torch.as_tensor(value, dtype=torch.long) for key, value in selected.items()}


def side_width_height(x: torch.Tensor, sides: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    width = x[sides["right"], 0].mean() - x[sides["left"], 0].mean()
    height = x[sides["top"], 1].mean() - x[sides["bottom"], 1].mean()
    return width, height


def rollout_physical_positions(bundle: dict, reference, stiffness: torch.Tensor) -> torch.Tensor:
    """Autonomous frame-0 to frame-199 prediction from a singleton reference."""
    edited = stiffness_graph(reference, stiffness)
    sim = [edited]
    params = bundle["params"]
    ae, dyn = bundle["ae"], bundle["dyn"]
    z = encode_frame_latent(
        ae, sim, 0, pos_dim=2, node_feature_mode=params["node_feature_mode"],
        normalizers=bundle["normalizers"], device="cpu",
    )
    context = encode_reference_context(
        ae, sim, pos_dim=2, normalizers=bundle["normalizers"], device="cpu",
        pool_mode="mean", include_temperature=False, include_source_id=False,
    )
    for _ in range(199):
        z = latent_step(dyn, z, bundle["latent_stats"], loss_mode="delta", context=context)
    x_normalized = decode_latent_positions(
        ae, sim, z, 199, pos_dim=2, ae_target_mode=params["ae_target_mode"],
        normalizers=bundle["normalizers"], device="cpu",
    )
    half = edited.reference_box_half_extent[:2].to(x_normalized)
    center = edited.reference_box_center[:2].to(x_normalized)
    return x_normalized.double() * half.double() + center.double()


def objective(bundle, reference, k0, active, x0, sides, scale_x, u):
    predicted = rollout_physical_positions(bundle, reference, k0 * u.exp())
    width0, height0 = side_width_height(x0, sides)
    width1, height1 = side_width_height(predicted, sides)
    dx, dy = width1 - width0, height1 - height0
    safe_dx = dx + EPS
    safe_dy = dy + EPS
    first = -dy / safe_dx
    second = -dx / safe_dy
    predicted_p = torch.where(first.abs() < second.abs(), first, second)
    target_dx = (float(scale_x) - 1.0) * width0
    compression_error = ((dx - target_dx) / target_dx.abs().clamp_min(EPS)).square()
    loss = torch.relu(predicted_p + 0.15).square() + 0.001 * u[active].square().mean() + compression_error
    if not torch.isfinite(loss) or not torch.isfinite(predicted_p) or not torch.isfinite(dx):
        raise RuntimeError("Non-finite autonomous rollout objective.")
    return loss, predicted_p, dx, compression_error


def gradient_check(bundle, reference, k0, active, x0, sides, scale_x, *, seed, index):
    probe = torch.zeros_like(k0, requires_grad=True)
    loss, _, _, _ = objective(bundle, reference, k0, active, x0, sides, scale_x, probe)
    gradient = torch.autograd.grad(loss, probe)[0]
    direction = gradient.detach() * active
    direction[active] -= direction[active].mean()
    norm = direction.norm().clamp_min(EPS)
    direction = direction / norm
    with torch.no_grad():
        plus = objective(bundle, reference, k0, active, x0, sides, scale_x, 0.001 * direction)[0]
        minus = objective(bundle, reference, k0, active, x0, sides, scale_x, -0.001 * direction)[0]
    finite_difference = float((plus - minus) / 0.002)
    exact = float((gradient * direction).sum())
    relative = abs(finite_difference - exact) / max(abs(finite_difference), abs(exact), EPS)
    record = dict(model_seed=seed, index=index, autograd=exact,
                  finite_difference=finite_difference, relative_error=relative)
    if not np.isfinite(relative) or relative > 0.1:
        raise RuntimeError(f"Rollout gradient check failed: {record}")
    return record


def optimize(output: Path) -> None:
    def prohibit(event, args):
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork"}:
            raise RuntimeError("External process forbidden during ML optimization: " + event)

    sys.addaudithook(prohibit)
    torch.set_num_threads(4)
    torch.manual_seed(786)
    np.random.seed(786)
    prepared_path = PREP / SOURCE / "prepare/prepared.pt"
    prepared = torch.load(prepared_path, weights_only=False, map_location="cpu")
    bundle_paths, bundles = {}, {}
    for seed in SEEDS:
        path, bundle = frozen_bundle(seed)
        bundle_paths[seed], bundles[seed] = path, bundle
    recipe = dict(
        stage="optimize", status="running", job_id=os.environ.get("PBS_JOBID"),
        script_sha256=sha(__file__), rollout_code_root=str(ROLL_CODE),
        rollout_code_manifest_sha256=sha(ROLL_CODE / "hashes.json"),
        endpoint_code_manifest_sha256=sha(ENDPOINT_CODE / "hashes.json"),
        prepared_path=str(prepared_path), prepared_sha256=sha(prepared_path),
        preparation=json.loads((PREP / SOURCE / "prepare/preparation.json").read_text()),
        bundles={str(seed): dict(path=str(bundle_paths[seed]), sha256=sha(bundle_paths[seed])) for seed in SEEDS},
        source=SOURCE, indices=list(NETWORKS), model_seeds=list(SEEDS), bound=BOUND,
        adam_steps=STEPS, learning_rate=LR,
        objective="relu(predicted_physical_endpoint_p+0.15)^2 + 0.001*mean(active_log_edit^2) + relative_compression_dx_error^2",
        predicted_path="singleton normalized reference -> frozen AE -> frozen static context/delta MLP x199 -> decoder -> physical inversion",
        p_ratio="fixed original physical quantile-0.10 sides; min reciprocal endpoint branch",
        selection="minimum ML loss over pristine and 150 iterates; no physics or observed future state",
        controls="one signed-value permutation of each selected ML edit; preserves active values, zero sum, norm, and bound",
        verification="after all 15 graphs and edits are hashed, separate 120-step endpoint verification reports every design",
        external_processes_allowed=False, split_loading="suppressed; frozen weights/stats only; prepared frame-zero references",
        counts=dict(originals=3, ml_designs=6, matched_signed_permutation_controls=6, total=15),
    )
    dump(output / "recipe.json", recipe)
    records, checks = [], []
    for index in NETWORKS:
        reference, raw = prepared["references"][index], prepared["physical"][index]
        if str(getattr(reference, "coordinate_normalization", "")) != "position_normalization":
            raise ValueError("Prepared reference is not position normalized.")
        k0 = reference.edge_attr[:, 3].detach().clone()
        active = k0 > 0
        if not active.any() or (k0 < 0).any():
            raise ValueError("Expected nonnegative springs with at least one active edge.")
        sides = fixed_sides(raw)
        x0 = raw.x[:, :2].detach().clone().double()
        scale_x = float(prepared["scales"][index])
        folder = output / f"{SOURCE}_{index}"
        folder.mkdir(exist_ok=True)

        def save(label, u, *, kind, model_seed, extra):
            graph = physical_candidate(raw, reference, k0 * u.exp())
            if hasattr(graph, "registry_poisson_ratio"): graph.registry_poisson_ratio = None
            if not torch.equal(graph.x, raw.x) or not torch.equal(graph.edge_index, raw.edge_index):
                raise RuntimeError("Design changed geometry or topology.")
            if not torch.equal(graph.edge_attr[:, 3] == 0, raw.edge_attr[:, 3] == 0):
                raise RuntimeError("Design changed zero-spring support.")
            path, edit_path = folder / f"{label}.pt", folder / f"{label}_edit.pt"
            torch.save(graph, path)
            torch.save(u.detach().cpu(), edit_path)
            records.append(dict(source=SOURCE, index=index, variant=label, kind=kind,
                model_seed=model_seed, policy="rollout", bound=BOUND, scale_x=scale_x,
                file=str(path.resolve()), sha256=sha(path), edit_file=str(edit_path.resolve()),
                edit_sha256=sha(edit_path), max_abs_log_edit=float(u.abs().max()),
                mean_active_log_edit=float(u[active].mean()), **extra))

        save("original", torch.zeros_like(k0), kind="original", model_seed=None,
             extra=dict(ml_loss=None, predicted_p_ratio=None, predicted_dx=None, compression_error=None))
        for seed in SEEDS:
            bundle = bundles[seed]
            checks.append(gradient_check(bundle, reference, k0, active, x0, sides, scale_x, seed=seed, index=index))
            u = torch.nn.Parameter(torch.zeros_like(k0))
            optimizer = torch.optim.Adam([u], lr=LR)
            best_loss, best_u, best_info = float("inf"), None, None
            history = []
            for step in range(STEPS + 1):
                loss, predicted_p, dx, compression_error = objective(bundle, reference, k0, active, x0, sides, scale_x, u)
                row = dict(step=step, loss=float(loss.detach()), predicted_p_ratio=float(predicted_p.detach()),
                           predicted_dx=float(dx.detach()), compression_error=float(compression_error.detach()))
                history.append(row)
                if row["loss"] < best_loss:
                    best_loss, best_u, best_info = row["loss"], u.detach().clone(), row.copy()
                if step == STEPS:
                    break
                optimizer.zero_grad()
                loss.backward()
                if u.grad is None or not torch.isfinite(u.grad).all():
                    raise RuntimeError("Non-finite rollout stiffness gradient.")
                u.grad[active] -= u.grad[active].mean()
                u.grad[~active] = 0
                optimizer.step()
                with torch.no_grad():
                    project_centered_box_(u, active, BOUND)
            label = f"rollout_s{seed}"
            pd.DataFrame(history).to_csv(folder / f"{label}_history.csv", index=False)
            save(label, best_u, kind="ml", model_seed=seed, extra=dict(ml_loss=best_info["loss"],
                predicted_p_ratio=best_info["predicted_p_ratio"], predicted_dx=best_info["predicted_dx"],
                compression_error=best_info["compression_error"]))
            generator = torch.Generator().manual_seed(123 + index + seed)
            control = torch.zeros_like(best_u)
            ids = active.nonzero(as_tuple=False).flatten()
            control[ids] = best_u[ids][torch.randperm(len(ids), generator=generator)]
            if abs(float(control[active].mean())) > 2e-6:
                raise RuntimeError("Signed-permutation control lost zero-sum constraint.")
            save(f"rollout_s{seed}_permuted", control, kind="random", model_seed=seed,
                 extra=dict(ml_loss=None, predicted_p_ratio=None, predicted_dx=None,
                            compression_error=None, control_seed=123 + index + seed))
    if len(records) != 15 or sum(row["kind"] == "ml" for row in records) != 6:
        raise RuntimeError("Unexpected frozen-design count.")
    dump(output / "gradient_checks.json", checks)
    recipe.update(status="completed", max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(output / "recipe.json", recipe)
    dump(output / "frozen_designs.json", dict(recipe_sha256=sha(output / "recipe.json"),
         frozen_at_unix=time.time(), records=records))


def verify(output: Path) -> None:
    torch.set_num_threads(1)
    manifest = json.loads((output / "frozen_designs.json").read_text())
    if manifest["recipe_sha256"] != sha(output / "recipe.json"):
        raise RuntimeError("Frozen design recipe hash mismatch.")
    for record in manifest["records"]:
        if sha(record["file"]) != record["sha256"] or sha(record["edit_file"]) != record["edit_sha256"]:
            raise RuntimeError("Frozen design artifact hash mismatch.")
    spec = importlib.util.spec_from_file_location("rollout_endpoint_physics", ENDPOINT_HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    lammps = DONOR / ".runtime/lammps/bin/lmp"
    dump(output / "verification_recipe.json", dict(job_id=os.environ.get("PBS_JOBID"),
        manifest_sha256=sha(output / "frozen_designs.json"), script_sha256=sha(__file__),
        endpoint_helper=str(ENDPOINT_HELPER), endpoint_helper_sha256=sha(ENDPOINT_HELPER), lammps_sha256=sha(lammps), steps=120,
        feedback_to_optimizer=False, report_all_frozen_designs=True))
    rows = []
    for record in manifest["records"]:
        graph = torch.load(record["file"], weights_only=False, map_location="cpu")
        result = module.validate_endpoints({(record["index"], record["variant"]): graph},
            output / "physics" / record["source"], lammps,
            {record["index"]: record["scale_x"]}, compression_steps=120).iloc[0].to_dict()
        rows.append({**record, **result})
        pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    table = pd.DataFrame(rows)
    original = table.loc[table.kind == "original"].set_index(["source", "index"])["physical_p_ratio"]
    table["original_p_ratio"] = [original.loc[(row.source, row["index"])] for _, row in table.iterrows()]
    table["change"] = table["physical_p_ratio"] - table["original_p_ratio"]
    table.to_csv(output / "results.csv", index=False)
    dump(output / "verification_completed.json", dict(valid=int(table.status.eq("completed").sum()),
        total=len(table), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        manifest_sha256=sha(output / "frozen_designs.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("optimize", "verify"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "optimize":
        if (args.output / "recipe.json").exists():
            raise RuntimeError("Refuse to overwrite an optimization study.")
        optimize(args.output)
    else:
        verify(args.output)
