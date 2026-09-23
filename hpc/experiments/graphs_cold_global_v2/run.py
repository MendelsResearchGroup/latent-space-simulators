"""Frozen validation-only fractional/contact cold-rubber screening harness."""
from __future__ import annotations

import argparse, csv, hashlib, json, random, sys, types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DYNAMICS = ROOT / "src/lss/dynamics"
sys.path.insert(0, str(DYNAMICS))
# The repository's package initializer imports unrelated legacy modules with a
# bare ``graph_utils`` dependency.  Load only the frozen dynamics modules used
# here, while preserving the fractional simulator's qualified source import.
import global_context_simulator as _global_context_simulator
_lss = sys.modules.setdefault("lss", types.ModuleType("lss"))
_dynamics = sys.modules.setdefault("lss.dynamics", types.ModuleType("lss.dynamics"))
_dynamics.__path__ = [str(DYNAMICS)]
_lss.dynamics = _dynamics
sys.modules.setdefault("lss.dynamics.global_context_simulator", _global_context_simulator)
from cold_graphs import COLD_GRAPH_ROOT, load_cold_trajectory, position_p_ratios
from fractional_contact_simulator import (
    FractionalContactSimulator, build_interaction_graph, fractional_coordinates,
    minimum_image, unit_minimum_image,
)

DT = 200.0

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frame_data(frames):
    return [{key: getattr(frame, key) for key in ("pos", "edge_index", "edge_attr", "box_tensor", "atom_types", "time")} for frame in frames]

def fractional_velocity(previous, current):
    return unit_minimum_image(fractional_coordinates(current["pos"], current["box_tensor"]) - fractional_coordinates(previous["pos"], previous["box_tensor"])) / (current["time"] - previous["time"])

def raw_global(previous, current, initial):
    return torch.cat((initial["box_tensor"], current["box_tensor"], (current["box_tensor"] - previous["box_tensor"]) / (current["time"] - previous["time"])))

def raw_node(previous, current, initial):
    return torch.cat((fractional_coordinates(initial["pos"], initial["box_tensor"]), fractional_coordinates(current["pos"], current["box_tensor"]), fractional_velocity(previous, current)), dim=1)

def bond_stiffness(frame):
    return frame["edge_attr"][:, 4]

def raw_edge(current, contact_cutoff):
    return build_interaction_graph(current["pos"], current["edge_index"], current["box_tensor"], bond_stiffness(current), contact_cutoff=contact_cutoff)

def targets(previous, current, following):
    previous_fractional = fractional_coordinates(previous["pos"], previous["box_tensor"])
    current_fractional = fractional_coordinates(current["pos"], current["box_tensor"])
    following_fractional = fractional_coordinates(following["pos"], following["box_tensor"])
    fractional_acceleration = (unit_minimum_image(following_fractional - current_fractional) - unit_minimum_image(current_fractional - previous_fractional)) / DT**2
    yz_box_acceleration = (following["box_tensor"][1:] - 2 * current["box_tensor"][1:] + previous["box_tensor"][1:]) / DT**2
    return fractional_acceleration, yz_box_acceleration

def standardize(value, mean, std):
    return (value - mean) / std

def fit_normalizers(train, contact_cutoff, seed):
    rng = np.random.default_rng(seed)
    node_inputs, global_inputs, edge_inputs, node_targets, box_targets = [], [], [], [], []
    for frames in train:
        centers = rng.choice(np.arange(2, 399), size=16, replace=False)
        for center in centers:
            previous, current, following = frames[center - 1], frames[center], frames[center + 1]
            node, box = targets(previous, current, following)
            edge, _ = raw_edge(current, contact_cutoff)
            node_inputs.append(raw_node(previous, current, frames[0]))
            global_inputs.append(raw_global(previous, current, frames[0]))
            edge_inputs.append(edge)
            node_targets.append(node)
            box_targets.append(box)
    def stats(values, stack=False):
        joined = torch.stack(values) if stack else torch.cat(values)
        return joined.mean(0), joined.std(0).clamp_min(1e-12)
    node_input_mean, node_input_std = stats(node_inputs)
    global_input_mean, global_input_std = stats(global_inputs, stack=True)
    edge_input_mean, edge_input_std = stats(edge_inputs)
    node_target_mean, node_target_std = stats(node_targets)
    box_target_mean, box_target_std = stats(box_targets, stack=True)
    return {"node_input_mean": node_input_mean, "node_input_std": node_input_std, "global_input_mean": global_input_mean, "global_input_std": global_input_std, "edge_input_mean": edge_input_mean, "edge_input_std": edge_input_std, "node_target_mean": node_target_mean, "node_target_std": node_target_std, "box_target_mean": box_target_mean, "box_target_std": box_target_std}

def model_inputs(previous, current, initial, normalizers, contact_cutoff):
    edge, edge_index = raw_edge(current, contact_cutoff)
    return (standardize(raw_node(previous, current, initial), normalizers["node_input_mean"], normalizers["node_input_std"]), current["atom_types"] - 1, standardize(edge, normalizers["edge_input_mean"], normalizers["edge_input_std"]), edge_index, standardize(raw_global(previous, current, initial), normalizers["global_input_mean"], normalizers["global_input_std"]))

def advance(model, previous, current, initial, normalizers, contact_cutoff):
    predicted_node, predicted_yz_box, _ = model(*model_inputs(previous, current, initial, normalizers, contact_cutoff))
    node_acceleration = predicted_node * normalizers["node_target_std"] + normalizers["node_target_mean"]
    yz_acceleration = predicted_yz_box * normalizers["box_target_std"] + normalizers["box_target_mean"]
    previous_fractional = fractional_coordinates(previous["pos"], previous["box_tensor"])
    current_fractional = fractional_coordinates(current["pos"], current["box_tensor"])
    next_fractional = current_fractional + unit_minimum_image(current_fractional - previous_fractional) + node_acceleration * DT**2
    # The source trajectories use a box-centred fractional convention.  Keep
    # autonomous states in the same [-0.5, 0.5) domain as the training inputs.
    next_fractional = unit_minimum_image(next_fractional)
    x_box = current["box_tensor"][0] + (current["box_tensor"][0] - previous["box_tensor"][0])
    yz_box = current["box_tensor"][1:] + (current["box_tensor"][1:] - previous["box_tensor"][1:]) + yz_acceleration * DT**2
    box = torch.cat((x_box.reshape(1), yz_box))
    if torch.any(box <= 0):
        raise ValueError("Predicted non-positive box length.")
    return {**current, "pos": next_fractional * box, "box_tensor": box, "time": current["time"] + DT}

def position_mse(prediction, truth):
    return minimum_image(prediction["pos"] - truth["pos"], truth["box_tensor"]).square().mean()

def physical_state_mse(prediction, truth):
    return position_mse(prediction, truth) + (prediction["box_tensor"][1:] - truth["box_tensor"][1:]).square().mean()

def normalized_state_loss(prediction, truth, normalizers):
    predicted_fractional = fractional_coordinates(prediction["pos"], prediction["box_tensor"])
    truth_fractional = fractional_coordinates(truth["pos"], truth["box_tensor"])
    fractional_scale = (normalizers["node_target_std"] * DT**2).clamp_min(1e-12)
    yz_box_scale = (normalizers["box_target_std"] * DT**2).clamp_min(1e-12)
    return (unit_minimum_image(predicted_fractional - truth_fractional) / fractional_scale).square().mean() + ((prediction["box_tensor"][1:] - truth["box_tensor"][1:]) / yz_box_scale).square().mean()

def sample_centers(ids, count, rng):
    values = [(identifier, int(center)) for identifier in ids for center in rng.choice(np.arange(2, 399), size=count, replace=False)]
    rng.shuffle(values)
    return values

def sample_windows(ids, count, unroll, rng):
    values = [(identifier, int(start)) for identifier in ids for start in rng.choice(np.arange(2, 400 - unroll), size=count, replace=False)]
    rng.shuffle(values)
    return values

def rollout(model, frames, horizon, normalizers, contact_cutoff):
    previous, current = frames[1], frames[2]
    for _ in range(3, horizon + 1):
        predicted = advance(model, previous, current, frames[0], normalizers, contact_cutoff)
        previous, current = current, predicted
    return current

def aligned_p_ratios(reference, positions, box):
    aligned = reference + minimum_image(positions - reference, box)
    return position_p_ratios(reference.numpy(), aligned.numpy())

def r2(rows, truth_key, prediction_key):
    pairs = [(row[truth_key], row[prediction_key]) for row in rows if np.isfinite(row[truth_key]) and np.isfinite(row[prediction_key])]
    if len(pairs) < 2: return None, len(pairs)
    truth, prediction = np.asarray(pairs, dtype=float).T
    denominator = np.square(truth - truth.mean()).sum()
    if denominator <= 0: return None, len(pairs)
    value = float(1.0 - np.square(prediction - truth).sum() / denominator)
    return (value if np.isfinite(value) else None), len(pairs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("fractional_bond", "fractional_contact"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    split_path = (HERE / protocol["split_manifest"]).resolve()
    split = json.loads(split_path.read_text())
    required_splits = {"train", "validation", "final_test"}
    split_ids = [identifier for key in required_splits for identifier in split[key]]
    if not required_splits.issubset(split) or len(split["train"]) != 48 or len(split["validation"]) != 12 or len(split["final_test"]) != 12 or len(set(split_ids)) != 72:
        raise ValueError("Expected the frozen disjoint 48/12/12 v2 manifest.")
    if args.seed != 123: raise ValueError("This frozen screen permits seed 123 only.")
    if args.smoke:
        train_ids, validation_ids, evaluation_ids = split["train"][:1], split["validation"][:1], split["validation"][1:2]
        phase1_epochs, phase2_epochs, centers, windows, unroll = 1, 1, 2, 1, 2
        evaluation_label = "smoke_validation_1"
    else:
        train_ids, validation_ids, evaluation_ids = split["train"], split["validation"], split["validation"]
        phase1_epochs, phase2_epochs = protocol["phase1"]["epochs"], protocol["phase2"]["max_epochs"]
        centers, windows, unroll = protocol["phase1"]["centers_per_train_trajectory_epoch"], protocol["phase2"]["windows_per_train_trajectory_epoch"], protocol["phase2"]["unroll_steps"]
        evaluation_label = "validation"
    if set(evaluation_ids) - set(split["validation"]): raise RuntimeError("This harness is validation-only and must never load final_test.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    fit_ids = list(dict.fromkeys(train_ids + validation_ids))
    loaded = {identifier: frame_data(load_cold_trajectory(identifier)) for identifier in fit_ids}
    variant = protocol["variants"][args.variant]; cutoff = variant["contact_cutoff"]
    normalizers = fit_normalizers([loaded[x] for x in train_ids], cutoff, protocol["normalization"]["seed"])
    architecture = protocol["architecture"]
    model = FractionalContactSimulator(**architecture, recurrent=variant["recurrent"])
    rng = np.random.default_rng(args.seed)
    validation_rng = np.random.default_rng(protocol["phase1"]["validation_seed"])
    validation_centers = {
        identifier: validation_rng.choice(
            np.arange(2, 399),
            size=(2 if args.smoke else protocol["phase1"]["validation_centers_per_trajectory"]),
            replace=False,
        )
        for identifier in validation_ids
    }
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["phase1"]["learning_rate"], weight_decay=protocol["phase1"]["weight_decay"])
    best_phase1, best_state = float("inf"), None
    for epoch in range(phase1_epochs):
        model.train(); optimizer.zero_grad(); losses = []
        samples = sample_centers(train_ids, centers, rng)
        for number, (identifier, center) in enumerate(samples, 1):
            frames = loaded[identifier]; node_true, box_true = targets(frames[center - 1], frames[center], frames[center + 1])
            node_pred, box_pred, _ = model(*model_inputs(frames[center - 1], frames[center], frames[0], normalizers, cutoff))
            loss = (node_pred - standardize(node_true, normalizers["node_target_mean"], normalizers["node_target_std"])).square().mean() + (box_pred - standardize(box_true, normalizers["box_target_mean"], normalizers["box_target_std"])).square().mean()
            (loss / protocol["phase1"]["gradient_accumulation"]).backward(); losses.append(float(loss.detach()))
            if number % protocol["phase1"]["gradient_accumulation"] == 0 or number == len(samples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["phase1"]["gradient_clip_norm"]); optimizer.step(); optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            validation = float(np.mean([
                physical_state_mse(
                    advance(model, loaded[identifier][t - 1], loaded[identifier][t], loaded[identifier][0], normalizers, cutoff),
                    loaded[identifier][t + 1],
                ).item()
                for identifier in validation_ids
                for t in validation_centers[identifier]
            ]))
        history.append({"phase": "one_step", "epoch": epoch + 1, "training_standardized_acceleration_mse": float(np.mean(losses)), "validation_physical_state_mse": validation})
        if validation < best_phase1: best_phase1, best_state = validation, {name: value.detach().clone() for name, value in model.state_dict().items()}
    if best_state is None: raise RuntimeError("No phase-1 checkpoint.")
    model.load_state_dict(best_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["phase2"]["learning_rate"], weight_decay=protocol["phase2"]["weight_decay"])
    best_phase2, best_phase2_state, stale = float("inf"), None, 0
    selection_horizons = (4,) if args.smoke else tuple(protocol["phase2"]["selection_horizons"])
    for epoch in range(phase2_epochs):
        model.train(); losses = []
        for identifier, start in sample_windows(train_ids, windows, unroll, rng):
            frames = loaded[identifier]; previous, current = frames[start - 1], frames[start]; loss = torch.zeros(())
            for offset in range(1, unroll + 1):
                predicted = advance(model, previous, current, frames[0], normalizers, cutoff)
                loss = loss + normalized_state_loss(predicted, frames[start + offset], normalizers)
                previous, current = current, predicted
            loss = loss / unroll; optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["phase2"]["gradient_clip_norm"]); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = float(np.mean([physical_state_mse(rollout(model, loaded[x], horizon, normalizers, cutoff), loaded[x][horizon]).item() for x in validation_ids for horizon in selection_horizons]))
        history.append({"phase": "autonomous", "epoch": epoch + 1, "training_normalized_fractional_position_yz_box_state_loss": float(np.mean(losses)), "validation_autonomous_physical_state_mse": validation, "selection_horizons": list(selection_horizons)})
        if validation < best_phase2:
            best_phase2, best_phase2_state, stale = validation, {name: value.detach().clone() for name, value in model.state_dict().items()}, 0
        else: stale += 1
        if stale >= protocol["phase2"]["patience"]: break
    if best_phase2_state is None: raise RuntimeError("No phase-2 checkpoint.")
    model.load_state_dict(best_phase2_state); args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_phase2_state, "normalizers": normalizers}, args.output / "checkpoint.pt")
    with (args.output / "training_history.csv").open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(history)
    rows = []
    model.eval()
    with torch.no_grad():
        for identifier in evaluation_ids:
            # In the smoke gate, the evaluation trajectory is intentionally not
            # loaded until both training phases and checkpoint selection finish.
            frames = loaded.get(identifier)
            if frames is None:
                frames = frame_data(load_cold_trajectory(identifier))
            previous, current = frames[1], frames[2]
            for frame in range(3, 400):
                predicted = advance(model, previous, current, frames[0], normalizers, cutoff)
                pred_y, pred_z = aligned_p_ratios(frames[0]["pos"], predicted["pos"], predicted["box_tensor"])
                true_y, true_z = aligned_p_ratios(frames[0]["pos"], frames[frame]["pos"], frames[frame]["box_tensor"])
                rows.append({"network_id": identifier, "frame": frame, "position_mse_physical_minimum_image": position_mse(predicted, frames[frame]).item(), "pred_p_y_over_x": pred_y, "true_p_y_over_x": true_y, "pred_p_z_over_x": pred_z, "true_p_z_over_x": true_z})
                previous, current = current, predicted
    with (args.output / "per_network_per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"variant": args.variant, "seed": args.seed, "smoke": args.smoke, "evaluation_split": evaluation_label, "best_phase1_validation_one_step_physical_state_mse": best_phase1, "best_phase2_validation_autonomous_physical_state_mse": best_phase2, "phase2_epochs_completed": epoch + 1, "test": {}}
    for horizon in protocol["evaluation"]["horizons"]:
        selected = [row for row in rows if row["frame"] == horizon]; y_r2, y_valid = r2(selected, "true_p_y_over_x", "pred_p_y_over_x"); z_r2, z_valid = r2(selected, "true_p_z_over_x", "pred_p_z_over_x")
        summary["test"][str(horizon)] = {"position_mse_physical_minimum_image": float(np.mean([row["position_mse_physical_minimum_image"] for row in selected])), "p_ratio_r2_y_over_x": y_r2, "p_ratio_r2_z_over_x": z_r2, "p_ratio_y_valid_total": [y_valid, len(selected)], "p_ratio_z_valid_total": [z_valid, len(selected)]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    data_ids = list(dict.fromkeys(fit_ids + evaluation_ids))
    recipe = {"runner_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(HERE / "protocol.json"), "split_manifest_sha256": sha256(split_path), "loader_sha256": sha256(ROOT / "src/lss/dynamics/cold_graphs.py"), "model_source_sha256": sha256(ROOT / "src/lss/dynamics/fractional_contact_simulator.py"), "cohort_sha256": sha256(COLD_GRAPH_ROOT / "clean_400.json"), "trajectory_sha256": {identifier: sha256(COLD_GRAPH_ROOT / f"{identifier}.pt") for identifier in data_ids}, "variant": args.variant, "contact_cutoff": cutoff, "recurrent": variant["recurrent"], "seed": args.seed, "smoke": args.smoke, "normalizer": {"train_ids": train_ids, **protocol["normalization"]}, "train_ids": train_ids, "validation_ids": validation_ids, "evaluation_split": evaluation_label, "evaluation_ids": evaluation_ids}
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")

if __name__ == "__main__": main()
