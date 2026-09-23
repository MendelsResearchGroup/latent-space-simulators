#!/usr/bin/env python3
"""Run paired physical-GNN arms on the canonical 60/20 noisy-LJ split.

The frozen shared latent cache deliberately has only 20 noisy-LJ training
trajectories.  This runner therefore reloads the canonical ljonly60 raw split
and encodes those exact graphs through each scope's frozen AE and matching
latent propagator before constructing augmented physical graphs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
CODE = Path(os.environ["LSS_CODE_ROOT"]).resolve()
OUT = ROOT / "notebooks/results/lj_mp2_scope20"
LEGACY = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2"
PILOT = ROOT / "notebooks/results/lj_latent_influence_edges/code_v2/run.py"
SCOPES = ("ljonly60", "shared3x20")
ARMS = ("base", "random", "latent")
SEEDS = (123, 456, 786)
HORIZONS = (10, 20, 30, 40, 50, 75, 100)

# Load the compact AE package first. Its cache loader needs its matching
# dataset resolver, which shares the lss.data module name with the older pilot.
sys.path.insert(0, str(LEGACY / "src"))
import lss
from lss.latent.experiment import _load_ae_cache
from lss.latent.training import compact_reference_edge_features, encode_frame_latent, reference_positions_for_model
from lss.graph import clone_graph
from graph_utils.box import Box

# The physical GNN and perturbation-edge rule are frozen in the prior pilot.
lss.__path__.append(str(PILOT.parent / "src" / "lss"))
lss.__path__.append(str(ROOT / "notebooks/results/lj100_broad/code_v1/src/lss"))
active_code_root = os.environ["LSS_CODE_ROOT"]
os.environ["LSS_CODE_ROOT"] = str(PILOT.parent)
pilot_spec = importlib.util.spec_from_file_location("frozen_edge_pilot", PILOT)
assert pilot_spec is not None and pilot_spec.loader is not None
pilot = importlib.util.module_from_spec(pilot_spec)
pilot_spec.loader.exec_module(pilot)
os.environ["LSS_CODE_ROOT"] = active_code_root
_select_topk = pilot.select_topk_nonphysical_edges
pilot.select_topk_nonphysical_edges = lambda scores, base_edges, _k: _select_topk(scores, base_edges, 5)
pilot.HORIZONS = HORIZONS


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scope(scope: str, seed: int) -> tuple[dict, dict, dict, Path, Path]:
    """Load one frozen AE and its matching fitted latent propagator."""
    checkpoint = OUT / f"{scope}_s{seed}" / "full" / "ae.pt"
    recipe_path = checkpoint.parent / "recipe.json"
    ae_done = json.loads((checkpoint.parent / "completed.json").read_text())
    expected = json.loads((CODE / "downstream_protocol.json").read_text())["frozen_autoencoders"][scope][str(seed)]
    assert ae_done["status"] == "completed" and sha(checkpoint) == expected["sha256"]
    ae_result = _load_ae_cache(checkpoint, json.loads(recipe_path.read_text())["config"], device="cpu")
    latent_dir = OUT / "latent" / f"{scope}_s{seed}" / "full"
    latent_done = json.loads((latent_dir / "completed.json").read_text())
    cache_path, model_path = latent_dir / "cache.pt", latent_dir / "latent_model.pt"
    assert latent_done["status"] == "completed"
    assert sha(cache_path) == latent_done["cache_sha256"] and sha(model_path) == latent_done["latent_model_sha256"]
    latent_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    latent_bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    assert latent_bundle["family"] == "history_attention" and latent_bundle["context_mode"] == "mean"
    assert int(latent_bundle["latent_dim"]) == 2
    model = pilot.make_trajectory_propagator("history_attention", "mean", latent_dim=2,
                                             token_dim=int(latent_bundle["token_dim"]))
    model.load_state_dict(latent_bundle["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return ae_result, latent_cache, {"model": model, "cache": cache_path, "model_path": model_path}, checkpoint, latent_dir


def canonical_raw(seed: int) -> tuple[list, list, Path]:
    """Reload the authoritative 60/20 LJ split; never use shared-cache rows."""
    checkpoint = OUT / f"ljonly60_s{seed}" / "full" / "ae.pt"
    recipe = json.loads((checkpoint.parent / "recipe.json").read_text())
    result = _load_ae_cache(checkpoint, recipe["config"], device="cpu")
    train, validation = result["train_data"], result["val_data"]
    train_ids = [int(sim[0].source_sim_id) for sim in train]
    val_ids = [int(sim[0].source_sim_id) for sim in validation]
    assert len(train) == 60 and len(validation) == 20 and not result["test_data"]
    assert Counter(str(sim[0].source_name) for sim in train + validation) == {"lj_noisy": 80}
    assert len(set(train_ids)) == 60 and len(set(val_ids)) == 20 and not set(train_ids).intersection(val_ids)
    return train, validation, checkpoint


def physical_graph(graph):
    """Restore physical coordinates while retaining frozen augmented topology."""
    out = clone_graph(graph)
    center = graph.reference_box_center.float()
    half = graph.reference_box_half_extent.float()
    out.x = out.x.clone()
    out.x[:, :2] = graph.x[:, :2].float() * half + center
    box = graph.box
    out.box = Box(
        float(box.x1 * half[0] + center[0]),
        float(box.x2 * half[0] + center[0]),
        float(box.y1 * half[1] + center[1]),
        float(box.y2 * half[1] + center[1]),
        -0.1,
        0.1,
    )
    out.box_tensor = graph.box_tensor.float()[:2] * half
    out.coordinate_normalization = "physical"
    return out


def encode_records(simulations: list, split: str, ae_result: dict, latent_cache: dict, smoke: bool) -> list[dict]:
    ae, normalizers, params = ae_result["ae"], ae_result["normalizers"], ae_result["params"]
    # The frozen edge builder asserts the complete canonical 60/20 split before
    # applying its smoke edge limits, so smoke only shortens GNN fitting.
    limit = 60 if split == "train" else 20
    records = []
    for simulation in simulations[:limit]:
        reference_graph = simulation[0]
        reference = reference_graph.x[:, :2].float()
        span = (reference.amax(0) - reference.amin(0)).clamp_min(1e-6)
        reference_edges = compact_reference_edge_features(reference_graph, pos_dim=2, device="cpu")
        normalized_edges = (reference_edges - normalizers["ref_edge_mean"]) / normalizers["ref_edge_std"]
        h_ref = ae.encode_reference_graph(reference_positions_for_model(reference_graph, pos_dim=2, device="cpu"),
                                          normalized_edges, reference_graph.edge_index.long()).detach()
        latent_raw = torch.stack([encode_frame_latent(ae, simulation, frame, pos_dim=2,
                                  node_feature_mode=params["node_feature_mode"], normalizers=normalizers,
                                  device="cpu") for frame in range(101)]).detach()
        physical = torch.stack([frame.x[:, :2].float() for frame in simulation[:101]])
        center, half = reference_graph.reference_box_center.float(), reference_graph.reference_box_half_extent.float()
        physical = physical * half + center
        raw_reference = physical_graph(reference_graph)
        records.append({"source": "lj_noisy", "original_index": int(reference_graph.source_sim_id), "split": split,
                        "latent_norm": (latent_raw - latent_cache["latent_stats"]["z_mean"]) / latent_cache["latent_stats"]["z_std"],
                        "h_ref": h_ref, "reference_tokens": torch.cat((h_ref, reference), dim=-1).detach(),
                        "ref_pos": reference, "span": span, "half": half, "center": center,
                        "physical_positions": physical.detach(), "physical_reference": raw_reference})
    return records


def write_paths(output: Path, records: list[dict], predicted: list[torch.Tensor]) -> None:
    """Persist all physical paths so post-fit metrics can be recomputed later."""
    rows = []
    for record, path in zip(records, predicted):
        source = record["row"]
        rows.append({"original_index": record["original_index"], "reference": source["physical_reference"],
                     "true_positions_physical": source["physical_positions"],
                     "predicted_positions_physical": path.double() * source["half"] + source["center"]})
    torch.save({"frames": list(range(101)), "positions_coordinate_system": "physical",
                "records": rows, "postfit_horizons": list(HORIZONS)}, output / "predicted_position_paths.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(8)
    protocol = json.loads((CODE / "downstream_protocol.json").read_text())
    manifest = json.loads((CODE / "downstream_manifest.json").read_text())
    assert {name: sha(CODE / name) for name in manifest} == manifest
    ae_result, latent_cache, frozen, ae_path, latent_dir = load_scope(args.scope, args.seed)
    train_raw, validation_raw, canonical_path = canonical_raw(args.seed)
    ae, latent_model = ae_result["ae"], frozen["model"]
    frozen_before = {"ae": pilot.state_digest(ae.state_dict()), "latent": pilot.state_digest(latent_model.state_dict())}
    run_name = f"{args.scope}_{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else "")
    output = OUT / "downstream" / "runs" / run_name
    output.mkdir(parents=True, exist_ok=False)
    recipe = {"scope": args.scope, "arm": args.arm, "seed": args.seed, "smoke": args.smoke,
              "job_id": os.environ.get("PBS_JOBID", "local"), "protocol": protocol,
              "protocol_sha256": sha(CODE / "downstream_protocol.json"), "code_manifest_sha256": sha(CODE / "downstream_manifest.json"),
              "canonical_lj_ae_checkpoint": str(canonical_path), "canonical_lj_ae_sha256": sha(canonical_path),
              "scope_ae_checkpoint": str(ae_path), "scope_ae_sha256": sha(ae_path),
              "latent_cache": str(frozen["cache"]), "latent_cache_sha256": sha(frozen["cache"]),
              "latent_model": str(frozen["model_path"]), "latent_model_sha256": sha(frozen["model_path"]),
              "final_test_used": False, "response_supervision_or_selection": False,
              "validation_selection": "one-step normalized acceleration MSE", "frozen_before": frozen_before}
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    try:
        train_rows = encode_records(train_raw, "train", ae_result, latent_cache, args.smoke)
        val_rows = encode_records(validation_raw, "validation", ae_result, latent_cache, args.smoke)
        assert len(train_rows) == 60 and len(val_rows) == 20
        train_records, validation_records = pilot.prepare_records({"train": train_rows, "validation": val_rows,
                                                                      "normalizers": ae_result["normalizers"],
                                                                      "latent_stats": latent_cache["latent_stats"]}, ae, latent_model, args.arm, args.smoke)
        normalizers = pilot.fit_normalizers(train_records)
        model = pilot.NormalizedAccelerationGNN(hidden_dim=64, steps=4)
        state, history = pilot.train(model, train_records, validation_records, normalizers, args.smoke, args.seed)
        model.load_state_dict(state)
        predicted = pilot.batched_rollout(model, validation_records, normalizers)
        table = pilot.evaluate_rollout(validation_records, predicted, f"{args.scope}_{args.arm}", args.seed)
        summary = pilot.summarize(table)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        table.to_csv(output / "per_network_rows.csv", index=False); summary.to_csv(output / "summary.csv", index=False)
        write_paths(output, validation_records, predicted)
        torch.save({"state_dict": state, "normalizers": normalizers}, output / "model.pt")
        frozen_after = {"ae": pilot.state_digest(ae.state_dict()), "latent": pilot.state_digest(latent_model.state_dict())}
        assert frozen_before == frozen_after
        recipe.update({"frozen_after": frozen_after, "checkpoint_sha256": sha(output / "model.pt"),
                       "predicted_position_paths_sha256": sha(output / "predicted_position_paths.pt"), "summary": summary.to_dict("records")})
        (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        (output / "completed.json").write_text(json.dumps({"status": "completed", "job_id": recipe["job_id"]}, indent=2) + "\n")
        print(summary.to_string(index=False), flush=True)
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
