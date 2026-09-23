"""Screen whether the first observed displacement predicts later cold-rubber response.

This is a validation-only diagnostic.  It fits box displacement from positions,
graph structure, and loading context; p-ratio is used only after fitting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DYNAMICS = ROOT / "src/lss/dynamics"
sys.path.insert(0, str(DYNAMICS))

import global_context_simulator as _global_context_simulator

_lss = sys.modules.setdefault("lss", types.ModuleType("lss"))
_dynamics = sys.modules.setdefault("lss.dynamics", types.ModuleType("lss.dynamics"))
_dynamics.__path__ = [str(DYNAMICS)]
_lss.dynamics = _dynamics
sys.modules.setdefault("lss.dynamics.global_context_simulator", _global_context_simulator)

from cold_graphs import load_cold_trajectory
from fractional_contact_simulator import fractional_coordinates, unit_minimum_image


V2_RUNNER = ROOT / "hpc/experiments/graphs_cold_global_v2/run.py"
SPLIT = ROOT / "hpc/experiments/graphs_cold_global/split_manifest_v2.json"
_spec = importlib.util.spec_from_file_location("cold_v2_screen_base", V2_RUNNER)
base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(base)

HORIZONS = np.arange(3, 400)
REPORT_HORIZONS = (20, 50, 100, 200, 399)


def frame_data(identifier: str) -> list[dict[str, torch.Tensor]]:
    frames = load_cold_trajectory(identifier)
    return [
        {
            key: getattr(frame, key)
            for key in ("pos", "edge_index", "edge_attr", "box_tensor", "atom_types", "time")
        }
        for frame in frames
    ]


def structure_features(frames: list[dict[str, torch.Tensor]]) -> np.ndarray:
    frame = frames[0]
    frac = fractional_coordinates(frame["pos"], frame["box_tensor"])
    stiffness = frame["edge_attr"][:, 4]
    lengths = frame["edge_attr"][:, 3]
    degrees = torch.bincount(frame["edge_index"].reshape(-1), minlength=len(frac)).float()
    covariance = torch.cov(frac.T)
    atom_fraction = (frame["atom_types"] == 2).float().mean()
    values = torch.cat(
        (
            frame["box_tensor"],
            torch.tensor([len(frac), frame["edge_index"].shape[1] / len(frac), atom_fraction]),
            frac.mean(0),
            frac.std(0),
            covariance[torch.triu_indices(3, 3)[0], torch.triu_indices(3, 3)[1]],
            torch.stack((lengths.mean(), lengths.std(), lengths.min(), lengths.max())),
            torch.stack((stiffness.mean(), stiffness.std(), stiffness.min(), stiffness.max())),
            torch.stack((degrees.mean(), degrees.std(), degrees.min(), degrees.max())),
        )
    )
    return values.numpy().astype(np.float64)


def early_features(frames: list[dict[str, torch.Tensor]]) -> tuple[np.ndarray, np.ndarray]:
    zero, one = frames[0], frames[1]
    dt = float(one["time"] - zero["time"])
    box_velocity = ((one["box_tensor"] - zero["box_tensor"]) / dt).numpy()
    delta = unit_minimum_image(
        fractional_coordinates(one["pos"], one["box_tensor"])
        - fractional_coordinates(zero["pos"], zero["box_tensor"])
    ) / dt
    node_summary = torch.cat(
        (
            delta.mean(0),
            delta.std(0),
            delta.square().mean(0).sqrt(),
            torch.cov(delta.T)[torch.triu_indices(3, 3)[0], torch.triu_indices(3, 3)[1]],
        )
    ).numpy()
    return box_velocity.astype(np.float64), node_summary.astype(np.float64)


def horizon_basis(frames: list[dict[str, torch.Tensor]], horizon: int) -> np.ndarray:
    initial = frames[0]["box_tensor"].numpy().astype(np.float64)
    x = float(frames[0]["box_tensor"][0] + horizon * (frames[1]["box_tensor"][0] - frames[0]["box_tensor"][0]))
    axial = (x - initial[0]) / initial[0]
    return np.asarray([1.0, axial, axial**2, axial**3], dtype=np.float64)


def rows(frames_by_id: dict[str, list[dict[str, torch.Tensor]]], identifiers: list[str], kind: str):
    design, targets, meta = [], [], []
    for identifier in identifiers:
        frames = frames_by_id[identifier]
        static = structure_features(frames)
        early_box, early_node = early_features(frames)
        if kind == "loading":
            graph = np.ones(1)
        elif kind == "static":
            graph = np.concatenate((np.ones(1), static))
        elif kind == "early_box":
            graph = np.concatenate((np.ones(1), static, early_box))
        elif kind == "early_node":
            graph = np.concatenate((np.ones(1), static, early_node))
        elif kind == "all":
            graph = np.concatenate((np.ones(1), static, early_box, early_node))
        else:
            raise ValueError(kind)
        for horizon in HORIZONS:
            design.append(np.kron(horizon_basis(frames, int(horizon)), graph))
            targets.append((frames[int(horizon)]["box_tensor"][1:] - frames[0]["box_tensor"][1:]).numpy())
            meta.append((identifier, int(horizon)))
    return np.asarray(design), np.asarray(targets), meta


def fit_ridge(design: np.ndarray, targets: np.ndarray, alpha: float):
    mean = design.mean(0)
    std = design.std(0)
    std[std < 1e-12] = 1.0
    normalized = (design - mean) / std
    normalized = np.column_stack((np.ones(len(normalized)), normalized))
    penalty = np.eye(normalized.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(normalized.T @ normalized + penalty, normalized.T @ targets)
    return mean, std, coefficients


def predict(design: np.ndarray, fit) -> np.ndarray:
    mean, std, coefficients = fit
    normalized = np.column_stack((np.ones(len(design)), (design - mean) / std))
    return normalized @ coefficients


def evaluate(frames_by_id, meta, predicted_displacement, horizons):
    output = {}
    for horizon in horizons:
        rows = []
        for (identifier, frame), displacement in zip(meta, predicted_displacement):
            if frame != horizon:
                continue
            frames = frames_by_id[identifier]
            predicted_box = frames[0]["box_tensor"].clone()
            predicted_box[0] = frames[horizon]["box_tensor"][0]
            predicted_box[1:] += torch.as_tensor(displacement, dtype=predicted_box.dtype)
            predicted_positions = fractional_coordinates(frames[0]["pos"], frames[0]["box_tensor"]) * predicted_box
            py, pz = base.aligned_p_ratios(frames[0]["pos"], predicted_positions, predicted_box)
            ty, tz = base.aligned_p_ratios(frames[0]["pos"], frames[horizon]["pos"], frames[horizon]["box_tensor"])
            rows.append(
                {
                    "position_mse": float(base.minimum_image(predicted_positions - frames[horizon]["pos"], frames[horizon]["box_tensor"]).square().mean()),
                    "box_mse": float((predicted_box[1:] - frames[horizon]["box_tensor"][1:]).square().mean()),
                    "pred_y": py,
                    "true_y": ty,
                    "pred_z": pz,
                    "true_z": tz,
                }
            )
        yr2, yvalid = base.r2(rows, "true_y", "pred_y")
        zr2, zvalid = base.r2(rows, "true_z", "pred_z")
        output[str(horizon)] = {
            "position_mse": float(np.mean([row["position_mse"] for row in rows])),
            "yz_box_mse": float(np.mean([row["box_mse"] for row in rows])),
            "p_ratio_r2_y": yr2,
            "p_ratio_r2_z": zr2,
            "valid_total_y": [yvalid, len(rows)],
            "valid_total_z": [zvalid, len(rows)],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text())
    ids = split["train"] + split["validation"]
    frames_by_id = {identifier: frame_data(identifier) for identifier in ids}
    results = {
        "split": {"train": split["train"], "validation": split["validation"], "final_test_loaded": False},
        "alpha": args.alpha,
        "fit_target": "yz box displacement from frame 0",
        "selection": "none; fixed ridge alpha",
        "models": {},
    }
    for kind in ("loading", "static", "early_box", "early_node", "all"):
        train_x, train_y, _ = rows(frames_by_id, split["train"], kind)
        validation_x, _, validation_meta = rows(frames_by_id, split["validation"], kind)
        fitted = fit_ridge(train_x, train_y, args.alpha)
        results["models"][kind] = evaluate(
            frames_by_id, validation_meta, predict(validation_x, fitted), REPORT_HORIZONS
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
