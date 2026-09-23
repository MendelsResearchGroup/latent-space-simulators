"""Global cold-graph simulator trained through autonomous 20-step windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/lss/dynamics"))

from cold_graphs import load_cold_trajectory
from evaluate_periodic import aligned_p_ratios, prediction_r2
from global_context_simulator import LocalGlobalLocalSimulator
from run import DT, advance, frame_data, mi, mse_position, mse_state, norms


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_windows(ids: list[str], count: int, unroll: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    windows = [
        (network_id, int(start))
        for network_id in ids
        for start in rng.choice(np.arange(2, 400 - unroll), size=count, replace=False)
    ]
    rng.shuffle(windows)
    return windows


def rollout_to(model, normalizers, frames, horizon: int):
    previous, current = frames[1], frames[2]
    prediction = current
    for _frame in range(3, horizon + 1):
        prediction = advance(model, frames[0]["box_tensor"], frames[0], previous, current, normalizers)
        previous, current = current, prediction
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads((HERE / "protocol.json").read_text())
    protocol = json.loads((HERE / "protocol_multistep.json").read_text())
    split = json.loads((HERE / "split_manifest.json").read_text())
    if len(set(split["train"] + split["validation"] + split["final_test"])) != 24:
        raise ValueError("Split is not disjoint.")
    evaluation_ids = split["final_test"]
    evaluation_label = "final_test"
    if args.smoke:
        split = {"train": split["train"][:1], "validation": split["validation"][:1]}
        evaluation_ids = json.loads((HERE / "split_manifest.json").read_text())["validation"][1:2]
        evaluation_label = "smoke_eval"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fit_ids = list(dict.fromkeys(split["train"] + split["validation"]))
    loaded = {identifier: frame_data(load_cold_trajectory(identifier)) for identifier in fit_ids}
    normalizers = norms([loaded[identifier] for identifier in split["train"]])
    model = LocalGlobalLocalSimulator(
        node_feature_dim=9,
        edge_feature_dim=5,
        atom_type_count=2,
        hidden_dim=base["architecture"]["hidden_dim"],
        attention_heads=base["architecture"]["attention_heads"],
        message_steps_each_side=base["architecture"]["message_blocks_before"],
        attention_layers=base["models"]["global"]["attention_layers"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=protocol["training"]["learning_rate"],
        weight_decay=protocol["training"]["weight_decay"],
    )
    rng = np.random.default_rng(args.seed)
    unroll = 2 if args.smoke else protocol["training"]["unroll_steps"]
    windows_per_trajectory = 1 if args.smoke else protocol["training"]["windows_per_train_trajectory_epoch"]
    max_epochs = 1 if args.smoke else protocol["training"]["max_epochs"]
    node_scale = (normalizers["node_std"] * DT**2).clamp_min(1e-12)
    box_scale = (normalizers["box_std"] * DT**2).clamp_min(1e-12)
    best = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        losses = []
        windows = sample_windows(split["train"], windows_per_trajectory, unroll, rng)
        for network_id, start in windows:
            frames = loaded[network_id]
            previous, current = frames[start - 1], frames[start]
            loss = torch.zeros((), dtype=torch.float32)
            for offset in range(1, unroll + 1):
                predicted = advance(model, frames[0]["box_tensor"], frames[0], previous, current, normalizers)
                truth = frames[start + offset]
                node_error = mi(predicted["pos"] - truth["pos"], truth["box_tensor"]) / node_scale
                box_error = (predicted["box_tensor"] - truth["box_tensor"]) / box_scale
                loss = loss + node_error.square().mean() + box_error.square().mean()
                previous, current = current, predicted
            loss = loss / unroll
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["training"]["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        validation_errors = []
        selection_horizons = (20,) if args.smoke else (20, 50, 100)
        with torch.no_grad():
            for network_id in split["validation"]:
                frames = loaded[network_id]
                for horizon in selection_horizons:
                    validation_errors.append(mse_state(rollout_to(model, normalizers, frames, horizon), frames[horizon]))
        validation = float(np.mean(validation_errors))
        history.append({
            "epoch": epoch + 1,
            "training_autonomous_state_loss": float(np.mean(losses)),
            "validation_autonomous_state_mse": validation,
        })
        if validation < best:
            best = validation
            stale = 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= protocol["training"]["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No state-selected checkpoint was produced.")

    model.load_state_dict(best_state)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "normalizers": normalizers}, args.output / "checkpoint.pt")
    with (args.output / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    rows = []
    model.eval()
    with torch.no_grad():
        for network_id in evaluation_ids:
            frames = frame_data(load_cold_trajectory(network_id))
            previous, current = frames[1], frames[2]
            for frame in range(3, 400):
                predicted = advance(model, frames[0]["box_tensor"], frames[0], previous, current, normalizers)
                previous, current = current, predicted
                pred_y, pred_z = aligned_p_ratios(frames[0]["pos"], predicted["pos"], predicted["box_tensor"])
                true_y, true_z = aligned_p_ratios(frames[0]["pos"], frames[frame]["pos"], frames[frame]["box_tensor"])
                rows.append({
                    "network_id": network_id,
                    "frame": frame,
                    "position_mse_physical_minimum_image": mse_position(predicted, frames[frame]),
                    "pred_p_y_over_x": pred_y,
                    "true_p_y_over_x": true_y,
                    "pred_p_z_over_x": pred_z,
                    "true_p_z_over_x": true_z,
                })
    with (args.output / "per_network_per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": "global_multistep20",
        "seed": args.seed,
        "smoke": args.smoke,
        "evaluation_split": evaluation_label,
        "best_validation_autonomous_state_mse": best,
        "epochs_completed": epoch + 1,
        "test": {},
    }
    for horizon in base["test"]["horizons"]:
        horizon_rows = [row for row in rows if row["frame"] == horizon]
        y_r2, y_valid = prediction_r2(horizon_rows, "true_p_y_over_x", "pred_p_y_over_x")
        z_r2, z_valid = prediction_r2(horizon_rows, "true_p_z_over_x", "pred_p_z_over_x")
        summary["test"][str(horizon)] = {
            "position_mse_physical_minimum_image": float(np.mean([row["position_mse_physical_minimum_image"] for row in horizon_rows])),
            "p_ratio_r2_y_over_x": y_r2,
            "p_ratio_r2_z_over_x": z_r2,
            "p_ratio_y_valid_total": [y_valid, len(horizon_rows)],
            "p_ratio_z_valid_total": [z_valid, len(horizon_rows)],
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    source_root = Path("/rg/mendels_prj/s.sergey/graphs_cold")
    data_ids = list(dict.fromkeys(fit_ids + evaluation_ids))
    (args.output / "recipe.json").write_text(json.dumps({
        "runner_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(HERE / "protocol_multistep.json"),
        "base_protocol_sha256": sha256(HERE / "protocol.json"),
        "split_manifest_sha256": sha256(HERE / "split_manifest.json"),
        "loader_sha256": sha256(ROOT / "src/lss/dynamics/cold_graphs.py"),
        "model_source_sha256": sha256(ROOT / "src/lss/dynamics/global_context_simulator.py"),
        "cohort_sha256": sha256(source_root / "clean_400.json"),
        "trajectory_sha256": {identifier: sha256(source_root / f"{identifier}.pt") for identifier in data_ids},
        "seed": args.seed,
        "smoke": args.smoke,
        "train_ids": split["train"],
        "validation_ids": split["validation"],
        "evaluation_split": evaluation_label,
        "evaluation_ids": evaluation_ids,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
