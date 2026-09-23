"""Re-evaluate frozen rollouts with periodic coordinates aligned before affine fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "notebooks/results/graphs_cold_global"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/lss/dynamics"))

from cold_graphs import load_cold_trajectory, position_p_ratios
from global_context_simulator import LocalGlobalLocalSimulator
from run import advance, frame_data, mse_position


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned_p_ratios(reference: torch.Tensor, current: torch.Tensor, box: torch.Tensor) -> tuple[float, float]:
    """Align each periodic position to its nearest reference image, then fit strain."""
    displacement = current - reference
    displacement = displacement - torch.round(displacement / box) * box
    aligned = reference + displacement
    return position_p_ratios(reference.numpy(), aligned.numpy())


def prediction_r2(rows: list[dict], truth: str, prediction: str) -> tuple[float | None, int]:
    pairs = [(row[truth], row[prediction]) for row in rows if np.isfinite(row[truth]) and np.isfinite(row[prediction])]
    if len(pairs) < 2:
        return None, len(pairs)
    true, predicted = np.asarray(pairs, dtype=float).T
    denominator = np.square(true - true.mean()).sum()
    if denominator <= 0:
        return None, len(pairs)
    value = float(1.0 - np.square(predicted - true).sum() / denominator)
    return value if np.isfinite(value) else None, len(pairs)


def main() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    split = json.loads((HERE / "split_manifest.json").read_text())
    production = json.loads((RESULTS / "production_jobs.json").read_text())
    if sha256(HERE / "run.py") != production["runner_sha256"]:
        raise RuntimeError("The frozen rollout runner changed after production submission.")

    output_root = RESULTS / "periodic_evaluation"
    for model_name, model_recipe in protocol["models"].items():
        for seed in protocol["seeds"]:
            run_dir = RESULTS / "production" / model_name / str(seed)
            recipe = json.loads((run_dir / "recipe.json").read_text())
            if recipe["evaluation_ids"] != split["final_test"] or recipe["evaluation_split"] != "final_test":
                raise RuntimeError(f"Final-test contract mismatch in {run_dir}")
            checkpoint_path = run_dir / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = LocalGlobalLocalSimulator(
                node_feature_dim=9,
                edge_feature_dim=5,
                atom_type_count=2,
                hidden_dim=protocol["architecture"]["hidden_dim"],
                attention_heads=protocol["architecture"]["attention_heads"],
                message_steps_each_side=protocol["architecture"]["message_blocks_before"],
                attention_layers=model_recipe["attention_layers"],
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            rows = []
            with torch.no_grad():
                for network_id in split["final_test"]:
                    frames = frame_data(load_cold_trajectory(network_id))
                    previous, current = frames[1], frames[2]
                    for frame in range(3, 400):
                        predicted = advance(model, frames[0]["box_tensor"], frames[0], previous, current, checkpoint["normalizers"])
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
            output = output_root / model_name / str(seed)
            output.mkdir(parents=True, exist_ok=True)
            with (output / "per_network_per_frame.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = {"model": model_name, "seed": seed, "estimator": "nearest-reference periodic image, then independent affine coordinate fits", "test": {}}
            for horizon in protocol["test"]["horizons"]:
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
            (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            (output / "recipe.json").write_text(json.dumps({
                "checkpoint_sha256": sha256(checkpoint_path),
                "evaluator_sha256": sha256(Path(__file__)),
                "runner_sha256": production["runner_sha256"],
                "protocol_sha256": production["protocol_sha256"],
                "split_manifest_sha256": production["split_manifest_sha256"],
                "evaluation_split": "final_test",
                "evaluation_ids": split["final_test"],
                "model": model_name,
                "seed": seed,
            }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
