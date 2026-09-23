#!/usr/bin/env python3
"""Train matched physical GNN arms using frozen LJ-only latent edges."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
CODE = Path(os.environ["LSS_CODE_ROOT"]).resolve()
OUT = ROOT / "notebooks/results/ljonly_latent_influence_edges"
PREVIOUS = ROOT / "notebooks/results/lj_latent_influence_edges/code_v2/run.py"

previous_spec = importlib.util.spec_from_file_location("frozen_edge_pilot", PREVIOUS)
assert previous_spec is not None and previous_spec.loader is not None
pilot = importlib.util.module_from_spec(previous_spec)
previous_spec.loader.exec_module(pilot)
pilot.OUT = OUT
_select_topk = pilot.select_topk_nonphysical_edges
pilot.select_topk_nonphysical_edges = lambda scores, base_edges, _k: _select_topk(scores, base_edges, 5)


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_ae(data: dict):
    bundle = torch.load(data["ae_checkpoint"], map_location="cpu", weights_only=False)
    params, norm = bundle["params"], data["normalizers"]
    ae = pilot._autoencoder_class(params["autoencoder_model"])(
        pos_dim=2, node_feature_dim=norm["node_feature_mean"].numel(),
        edge_dim=norm["edge_mean"].numel(), hidden_size=params["hidden_size"],
        latent_dim=params["latent_dim"], latent_tokens=params["latent_tokens"],
        reconstruction_dim=norm["target_mean"].numel(),
        message_passing_steps=params.get("message_passing_steps", 0),
    )
    ae.set_edge_normalization(norm["edge_mean"], norm["edge_std"],
                              norm.get("ref_edge_mean"), norm.get("ref_edge_std"))
    ae.edge_mode = params["edge_mode"]
    incompatible = ae.load_state_dict(bundle["ae_state_dict"], strict=False)
    if incompatible.unexpected_keys or (params.get("message_passing_steps", 0) and incompatible.missing_keys):
        raise RuntimeError(f"Incompatible AE checkpoint: {incompatible}")
    ae.eval()
    for parameter in ae.parameters(): parameter.requires_grad_(False)
    return ae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("base", "random", "latent"), required=True)
    parser.add_argument("--seed", type=int, choices=(123, 456, 786), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    protocol = json.loads((CODE / "protocol.json").read_text())
    manifest = json.loads((CODE / "manifest.json").read_text())
    assert {name: sha(CODE / name) for name in manifest} == manifest
    latent_dir = OUT / "latent"
    latent_done = json.loads((latent_dir / "completed.json").read_text())
    assert latent_done["status"] == "completed"
    assert sha(latent_dir / "cache.pt") == latent_done["cache_sha256"]
    assert sha(latent_dir / "latent_model.pt") == latent_done["latent_model_sha256"]
    data = torch.load(latent_dir / "cache.pt", map_location="cpu", weights_only=False)
    assert len(data["train"]) == 60 and len(data["validation"]) == 20
    ae = load_ae(data)
    latent_model = pilot.make_trajectory_propagator("history_attention", "mean", latent_dim=4)
    latent_bundle = torch.load(latent_dir / "latent_model.pt", map_location="cpu", weights_only=False)
    latent_model.load_state_dict(latent_bundle["state_dict"], strict=True)
    latent_model.eval()
    for parameter in latent_model.parameters(): parameter.requires_grad_(False)
    frozen_before = {"ae": pilot.state_digest(ae.state_dict()),
                     "latent": pilot.state_digest(latent_model.state_dict())}
    name = f"{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else "")
    output = OUT / "runs" / name
    output.mkdir(parents=True, exist_ok=False)
    recipe = {"arm": args.arm, "seed": args.seed, "smoke": args.smoke,
              "job_id": os.environ.get("PBS_JOBID", "local"), "protocol": protocol,
              "protocol_sha256": sha(CODE / "protocol.json"),
              "code_manifest_sha256": sha(CODE / "manifest.json"),
              "latent_cache_sha256": sha(latent_dir / "cache.pt"),
              "latent_model_sha256": sha(latent_dir / "latent_model.pt"),
              "final_test_used": False, "response_supervision_or_selection": False,
              "validation_edge_inputs": "frame-0 reference graph and z0 only",
              "frozen_before": frozen_before}
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    try:
        train_records, validation_records = pilot.prepare_records(
            data, ae, latent_model, args.arm, args.smoke
        )
        normalizers = pilot.fit_normalizers(train_records)
        model = pilot.NormalizedAccelerationGNN(hidden_dim=64, steps=4)
        state, history = pilot.train(
            model, train_records, validation_records, normalizers, args.smoke, args.seed
        )
        model.load_state_dict(state)
        predictions = pilot.batched_rollout(model, validation_records, normalizers)
        table = pilot.evaluate_rollout(validation_records, predictions, args.arm, args.seed)
        summary = pilot.summarize(table)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        table.to_csv(output / "per_network_rows.csv", index=False)
        summary.to_csv(output / "summary.csv", index=False)
        torch.save({"state_dict": state, "normalizers": normalizers}, output / "model.pt")
        frozen_after = {"ae": pilot.state_digest(ae.state_dict()),
                        "latent": pilot.state_digest(latent_model.state_dict())}
        assert frozen_before == frozen_after
        recipe.update({"frozen_after": frozen_after, "checkpoint_sha256": sha(output / "model.pt"),
                       "summary": summary.to_dict("records")})
        (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        (output / "completed.json").write_text(json.dumps({"status": "completed",
            "job_id": recipe["job_id"]}, indent=2) + "\n")
        print(summary.to_string(index=False), flush=True)
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({"status": "failed",
            "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
