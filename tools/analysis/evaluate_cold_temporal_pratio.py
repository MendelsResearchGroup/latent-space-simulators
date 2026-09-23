"""Validation-only temporal-slope p-ratio diagnostic for saved cold rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/lss/dynamics"))

from cold_graphs import affine_strain, load_cold_trajectory


def slope(values: np.ndarray, frames: np.ndarray) -> float:
    design = np.column_stack((np.ones(len(frames)), frames))
    return float(np.linalg.lstsq(design, values, rcond=None)[0][1])


def r2(truth: list[float], prediction: list[float]) -> float | None:
    truth_array = np.asarray(truth)
    prediction_array = np.asarray(prediction)
    denominator = np.square(truth_array - truth_array.mean()).sum()
    if len(truth_array) < 2 or denominator <= 0:
        return None
    return float(1.0 - np.square(prediction_array - truth_array).sum() / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.rows.open() as handle:
        rows = list(csv.DictReader(handle))
    identifiers = sorted({row["network_id"] for row in rows})
    by_id = {identifier: [] for identifier in identifiers}
    for row in rows:
        by_id[row["network_id"]].append(row)
    loaded = {identifier: load_cold_trajectory(identifier) for identifier in identifiers}
    output = {"source_rows": str(args.rows), "estimator": "independent strain slopes versus frame with intercept", "windows": {}}
    maximum_frame = max(int(row["frame"]) for row in rows)
    for start, stop in ((3, 50), (3, 100), (3, 200), (3, 399), (50, 399), (100, 399), (200, 399)):
        if stop > maximum_frame:
            continue
        truth_y, truth_z, prediction_y, prediction_z = [], [], [], []
        for identifier in identifiers:
            selected = sorted(
                (row for row in by_id[identifier] if start <= int(row["frame"]) <= stop),
                key=lambda row: int(row["frame"]),
            )
            frame_indices = np.asarray([int(row["frame"]) for row in selected], dtype=float)
            trajectory = loaded[identifier]
            reference = trajectory[0].pos.numpy()
            true_strains = []
            for frame in frame_indices:
                current = trajectory[int(frame)]
                positions = current.pos.numpy()
                box = current.box_tensor.numpy()
                delta = positions - reference
                aligned = reference + delta - np.round(delta / box) * box
                true_strains.append(affine_strain(reference, aligned))
            true_strains = np.asarray(true_strains)
            true_x = true_strains[:, 0]
            pred_x = np.asarray([
                (float(trajectory[2].box_tensor[0]) + (int(frame) - 2) * (float(trajectory[2].box_tensor[0]) - float(trajectory[1].box_tensor[0])))
                / float(trajectory[0].box_tensor[0]) - 1.0
                for frame in frame_indices
            ])
            pred_y = -np.asarray([float(row["pred_p_y_over_x"]) for row in selected]) * pred_x
            pred_z = -np.asarray([float(row["pred_p_z_over_x"]) for row in selected]) * pred_x
            sx_true = slope(true_x, frame_indices)
            sx_pred = slope(pred_x, frame_indices)
            truth_y.append(-slope(true_strains[:, 1], frame_indices) / sx_true)
            truth_z.append(-slope(true_strains[:, 2], frame_indices) / sx_true)
            prediction_y.append(-slope(pred_y, frame_indices) / sx_pred)
            prediction_z.append(-slope(pred_z, frame_indices) / sx_pred)
        output["windows"][f"{start}:{stop}"] = {
            "p_ratio_r2_y": r2(truth_y, prediction_y),
            "p_ratio_r2_z": r2(truth_z, prediction_z),
            "valid_total_y": [len(truth_y), len(identifiers)],
            "valid_total_z": [len(truth_z), len(identifiers)],
            "true_y": truth_y,
            "predicted_y": prediction_y,
            "true_z": truth_z,
            "predicted_z": prediction_z,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
