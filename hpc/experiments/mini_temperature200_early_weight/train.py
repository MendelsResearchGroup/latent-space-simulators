#!/usr/bin/env python3
"""Run the matched 4:1 early-frame-weighted 200-MD AE/propagator pilot."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mini_temperature200_base_train", HERE / "base_train.py")
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_early_weighted(result: dict, out: Path, epochs: int) -> str:
    model, norm, params = result["ae"], result["normalizers"], result["params"]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    train, validation = result["train_data"], result["val_data"]
    names = sorted({str(simulation[0].source_name) for simulation in train})

    def batch(simulations, rows):
        return BASE.batch_delta_graphs(
            simulations, rows, pos_dim=2, device="cpu",
            node_feature_mode=params["node_feature_mode"], edge_mode=params["edge_mode"],
        )

    energy = {}
    with torch.no_grad():
        for name in names:
            indices = [index for index, simulation in enumerate(train)
                       if str(simulation[0].source_name) == name]
            values = []
            for frame in range(BASE.TRAIN_FRAMES):
                packed = batch(train, [(index, frame) for index in indices])
                target = BASE.ae_target_tensor(packed, params["ae_target_mode"]) / norm["target_std"]
                values.append(torch.stack([
                    target[packed["batch"] == graph_index].square().mean()
                    for graph_index in range(len(indices))
                ]).mean())
            energy[name] = torch.stack(values)

    raw = torch.ones(BASE.TRAIN_FRAMES, dtype=torch.float32)
    raw[:100] = 4.0
    raw /= raw.mean()
    assert torch.isfinite(raw).all() and bool((raw > 0).all())
    assert torch.isclose(raw.mean(), torch.tensor(1.0), atol=1e-7)
    assert torch.isclose(raw[0] / raw[-1], torch.tensor(4.0), atol=1e-7)
    weights = {name: raw.clone() for name in names}
    BASE.dump(out / "frame_weights.json", {
        name: {
            "energy_diagnostic": energy[name].tolist(),
            "weights": weights[name].tolist(),
            "post_warmup_weighting": "frames_0_99_four_times_frames_100_499_mean_normalized"
        }
        for name in names
    })

    def objective(simulations, rows):
        packed = batch(simulations, rows)
        target = (BASE.ae_target_tensor(packed, params["ae_target_mode"]) - norm["target_mean"]) / norm["target_std"]
        prediction, _ = model(
            (packed["node_feature"] - norm["node_feature_mean"]) / norm["node_feature_std"],
            packed["ref_pos"],
            (packed["edge_attr"] - norm["edge_mean"]) / norm["edge_std"],
            (packed["ref_edge_attr"] - norm["ref_edge_mean"]) / norm["ref_edge_std"],
            packed["edge_index"], packed["batch"],
        )
        error = (prediction - target).square().mean(-1)
        return torch.stack([
            error[packed["batch"] == graph_index].mean()
            * weights[str(simulations[index][0].source_name)][frame]
            for graph_index, (index, frame) in enumerate(rows)
        ]).mean()

    train_rows = [(index, frame) for index in range(len(train)) for frame in range(BASE.TRAIN_FRAMES)]

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        scores = []
        for name in names:
            rows = [
                (index, frame)
                for index, simulation in enumerate(validation)
                if str(simulation[0].source_name) == name
                for frame in range(BASE.TRAIN_FRAMES)
            ]
            total = 0.0
            for start in range(0, len(rows), 32):
                part = rows[start:start + 32]
                total += float(objective(validation, part)) * len(part)
            scores.append(total / len(rows))
        return float(np.mean(scores))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best, state, stale = validate(), copy.deepcopy(model.state_dict()), 0
    history = [{"epoch": 0, "val_loss": best}]
    for epoch in range(1, epochs + 1):
        random.shuffle(train_rows)
        model.train()
        total = 0.0
        for start in range(0, len(train_rows), 32):
            rows = train_rows[start:start + 32]
            loss = objective(train, rows)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        score = validate()
        history.append({"epoch": epoch, "train_loss": total / len(train_rows), "val_loss": score})
        pd.DataFrame(history).to_csv(out / "ae_history.csv", index=False)
        print("early4 AE", epoch, score, flush=True)
        if score < best - 1e-8:
            best, state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 12:
            break

    model.load_state_dict(state)
    bundle = torch.load(out / "ae.pt", map_location="cpu", weights_only=False)
    bundle["ae_state_dict"] = state
    bundle["ae_history"] = history
    bundle["frame_balance_recipe"] = {
        "training_only": True,
        "post_warmup_weighting": "frames_0_99_four_times_frames_100_499_mean_normalized",
        "early_to_late_ratio": 4.0,
        "warmup_epochs": 1,
        "validation_objective_matches_training": True,
        "response_used_for_training_or_selection": False,
    }
    torch.save(bundle, out / "ae.pt")
    return sha(out / "ae.pt")


def main() -> None:
    BASE.CODE = HERE
    BASE.fit_uniform = fit_early_weighted
    BASE.main()


if __name__ == "__main__":
    main()
