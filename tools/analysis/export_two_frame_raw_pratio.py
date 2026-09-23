"""Export raw frame-wise p-ratios from the frozen two-frame rollout study."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "notebooks/results/mixed_temperature_extrapolation_2frame"
CODE = STUDY / "code_v2"
OUTPUT = STUDY / "raw_pratio"
SOURCES = ("node_mixed", "stiffness_mixed")

os.environ["LSS_PROJECT_ROOT"] = str(ROOT)
os.environ["LSS_CODE_ROOT"] = str(CODE)
for path in (CODE, ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_study_module():
    spec = importlib.util.spec_from_file_location("two_frame_frozen_run", CODE / "run.py")
    if spec is None or spec.loader is None:
        raise ImportError(CODE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_p_ratio(width: np.ndarray, height: np.ndarray, width0: float, height0: float) -> np.ndarray:
    """Match graph_utils' endpoint convention using boundary-side changes."""
    dx = np.asarray(width, dtype=float) - width0
    dy = np.asarray(height, dtype=float) - height0
    with np.errstate(divide="ignore", invalid="ignore"):
        first = -dy / dx
        second = -dx / dy
    value = np.where(np.abs(first) < np.abs(second), first, second)
    value[(np.abs(dx) <= 1e-12) | (np.abs(dy) <= 1e-12)] = np.nan
    return value


def scalar_r2(truth: pd.Series, prediction: pd.Series) -> float:
    values = pd.concat((truth, prediction), axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        return float("nan")
    y = values.iloc[:, 0].to_numpy(float)
    yhat = values.iloc[:, 1].to_numpy(float)
    denominator = float(np.square(y - y.mean()).sum())
    return float(1.0 - np.square(y - yhat).sum() / denominator) if denominator > 0 else float("nan")


def evaluation_records(study, result: dict, protocol: dict) -> list[dict]:
    manifest = json.loads((STUDY / "data_manifest.json").read_text())
    identifiers = protocol["evaluation_ids"]
    records = []
    for source in SOURCES:
        path = manifest["datasets"][source]["test_path"]
        component = {
            "name": source, "label": source, "path": path,
            "train_count": 0, "val_count": len(identifiers),
            "split_indices": {"train": [], "val": identifiers, "test": []},
        }
        source_spec = {
            "name": source, "label": source, "path": path,
            "dataset_name": source, "source_name": source,
            "coordinate_normalization": "position_normalization",
            "dataset_mixture": [component],
        }
        _, simulations, _, _ = study.resolve_train_val_test(
            source_spec, result["params"], split_seed=123
        )
        observed = study.check_temperatures(simulations, protocol["evaluation_temperatures_K"])
        if not set(observed) <= set(protocol["evaluation_temperatures_K"]):
            raise AssertionError(observed)
        records.extend(study.encode_records(result, simulations, identifiers))
    return records


@torch.no_grad()
def export() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    study = load_study_module()
    protocol = json.loads((CODE / "protocol.json").read_text())
    result = study.load_ae(STUDY / "ae")
    checkpoint = torch.load(STUDY / "dynamics/dynamics.pt", map_location="cpu", weights_only=False)
    model = study.StochasticTwoFramePropagator().eval()
    model.load_state_dict(checkpoint["state_dict"])
    stats = checkpoint["stats"]
    records = evaluation_records(study, result, protocol)
    if len(records) != 162:
        raise AssertionError(f"Expected 162 evaluation trajectories, found {len(records)}")

    rows = []
    for record in records:
        simulation = record["simulation"]
        reference = simulation[0]
        physical_reference, _ = study._pref(reference)
        sides = study.directional_side_indices_from_box(physical_reference, quantile=0.10)
        reference_xy = physical_reference.x[:, :2].numpy()
        width0 = float(reference_xy[sides["right"], 0].mean() - reference_xy[sides["left"], 0].mean())
        height0 = float(reference_xy[sides["top"], 1].mean() - reference_xy[sides["bottom"], 1].mean())

        def width_height(graph) -> tuple[float, float]:
            xy = study.physical(graph, reference).x[:, :2].numpy()
            width = float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean())
            height = float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean())
            return width, height

        truth = np.asarray([
            (width0, height0) if frame == 0 else width_height(simulation[frame])
            for frame in range(200)
        ])
        paths = study.sample_paths(model, record, stats, samples=8)
        predicted = np.empty((len(paths), 200, 2), dtype=float)
        predicted[:, 0] = (width0, height0)
        predicted[:, 1] = truth[1]
        for sample_index, path in enumerate(paths):
            for frame in range(2, 200):
                graph = study.decode_latent_to_graph(
                    result["ae"], simulation, path[frame], frame, pos_dim=2,
                    ae_target_mode=result["params"]["ae_target_mode"],
                    normalizers=result["normalizers"], device="cpu",
                )
                predicted[sample_index, frame] = width_height(graph)

        true_p = raw_p_ratio(truth[:, 0], truth[:, 1], width0, height0)
        sample_p = np.stack([
            raw_p_ratio(sample[:, 0], sample[:, 1], width0, height0)
            for sample in predicted
        ])
        ensemble_mean = predicted.mean(axis=0)
        ensemble_p = raw_p_ratio(ensemble_mean[:, 0], ensemble_mean[:, 1], width0, height0)
        for frame in range(200):
            finite_samples = sample_p[:, frame][np.isfinite(sample_p[:, frame])]
            rows.append({
                "source": record["source"],
                "trajectory_id": int(record["trajectory_id"]),
                "temperature_K": float(record["temperature_K"]),
                "frame": frame,
                "true_p_ratio": float(true_p[frame]),
                "ensemble_mean_position_p_ratio": float(ensemble_p[frame]),
                "sample_p_ratio_mean": float(finite_samples.mean()) if len(finite_samples) else float("nan"),
                "sample_p_ratio_sd": float(finite_samples.std(ddof=1)) if len(finite_samples) > 1 else float("nan"),
                "valid_samples": int(len(finite_samples)),
                "samples": int(len(paths)),
                "observed": frame <= 1,
            })
        print("exported", record["source"], record["trajectory_id"], record["temperature_K"], flush=True)

    table = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "per_step_network.csv", index=False)
    table.query("frame==199").to_csv(OUTPUT / "final_network.csv", index=False)

    summaries = []
    groups = [(source, temperature, group) for (source, temperature), group in table.groupby(["source", "temperature_K"])]
    groups += [(source, "all_high_T", group) for source, group in table.groupby("source")]
    for source, temperature, group in groups:
        for frame, values in group.groupby("frame"):
            finite = values[["true_p_ratio", "ensemble_mean_position_p_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
            summaries.append({
                "source": source, "temperature_K": temperature, "frame": int(frame),
                "p_ratio_r2": scalar_r2(values.true_p_ratio, values.ensemble_mean_position_p_ratio),
                "p_ratio_mae": float(np.abs(finite.iloc[:, 0] - finite.iloc[:, 1]).mean()) if len(finite) else float("nan"),
                "valid": int(len(finite)), "total": int(len(values)),
            })
    pd.DataFrame(summaries).to_csv(OUTPUT / "summary_by_step.csv", index=False)
    metadata = {
        "status": "complete",
        "definition": "raw physical endpoint p-ratio at each frame using fixed frame-0 boundary sides; no trajectory fit",
        "prediction": "p-ratio of mean decoded physical positions across eight stochastic autonomous rollouts",
        "observed_frames": [0, 1], "forecast_frames": [2, 199],
        "evaluation_ids": protocol["evaluation_ids"], "reserved_ids": protocol["reserved_ids"],
        "checkpoint_sha256": sha256(STUDY / "dynamics/dynamics.pt"),
        "ae_sha256": sha256(STUDY / "ae/ae.pt"),
        "script_sha256": sha256(Path(__file__)),
        "final_test_used": False,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    export()
