"""Train on 20/30 K trajectories and evaluate frozen AE on paired/unseen low-T."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
HERE = Path(__file__).resolve().parent
OUT = ROOT / "notebooks/results/high_temperature_to_low_ae"
DONOR_CODE = ROOT / "notebooks/results/mixed_temperature_extrapolation_2frame/code_v2"
MANIFEST = ROOT / "notebooks/results/all4_coverage_1000/data_manifest.json"
os.environ.setdefault("LSS_CODE_ROOT", str(DONOR_CODE))

for path in (DONOR_CODE, ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from graph_utils import directional_side_indices_from_box
from lss.dynamics.experiment import resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.training import decode_latent_to_graph, encode_frame_latent


def module_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


study = module_from(DONOR_CODE / "run.py", "high_to_low_donor_run")
base_train = module_from(DONOR_CODE / "base_train.py", "high_to_low_base_train")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_spec(names: list[str], identifiers: list[int]) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    components = []
    for name in names:
        entry = manifest["datasets"][name]
        components.append({
            "name": name, "label": name, "path": entry["trainval_path"],
            "train_count": 0, "val_count": len(identifiers),
            "split_indices": {"train": [], "val": identifiers, "test": []},
        })
    return {
        "name": "high_to_low", "label": "high_to_low",
        "path": components[0]["path"], "dataset_name": "high_to_low",
        "source_name": "high_to_low", "coordinate_normalization": "position_normalization",
        "dataset_mixture": components,
    }


def training_spec(train_ids: list[int], validation_ids: list[int]) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    components = []
    for name in ("node_mixed", "stiffness_mixed"):
        path = manifest["datasets"][name]["trainval_path"]
        components.append({
            "name": name, "label": name, "path": path,
            "train_count": len(train_ids), "val_count": len(validation_ids),
            "split_indices": {"train": train_ids, "val": validation_ids, "test": []},
        })
    return {
        "name": "high_temperature_only", "label": "high_temperature_only",
        "path": components[0]["path"], "dataset_name": "high_temperature_only",
        "source_name": "high_temperature_only", "coordinate_normalization": "position_normalization",
        "dataset_mixture": components,
    }


def r2(truth, prediction) -> float:
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[finite], prediction[finite]
    denominator = np.square(truth - truth.mean()).sum()
    return float(1 - np.square(truth - prediction).sum() / denominator) if len(truth) > 1 and denominator > 0 else float("nan")


def fit_p(width, height, width0, height0) -> float:
    x = (np.asarray(width) - width0) / width0
    y = (np.asarray(height) - height0) / height0
    x, y = x[30:], y[30:]
    design = np.c_[np.ones(len(x)), x]
    return -float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def endpoint_p(width, height, width0, height0) -> float:
    dx, dy = float(width - width0), float(height - height0)
    if abs(dx) <= 1e-12 or abs(dy) <= 1e-12:
        return float("nan")
    first, second = -dy / dx, -dx / dy
    return float(first if abs(first) < abs(second) else second)


@torch.no_grad()
def evaluate(result: dict, identifiers: list[int], cohort: str, output: Path) -> None:
    spec = source_spec(["node_low", "stiffness_low"], identifiers)
    _, simulations, _, _ = resolve_train_val_test(spec, result["params"], split_seed=123)
    rows, latent_rows = [], []
    source_counts = {"node_low": 0, "stiffness_low": 0}
    for simulation in simulations:
        reference = simulation[0]
        source = str(reference.source_name)
        trajectory_id = int(identifiers[source_counts[source]])
        source_counts[source] += 1
        pref, _ = study._pref(reference)
        sides = directional_side_indices_from_box(pref, quantile=.10)
        xy0 = pref.x[:, :2].numpy()
        width0 = float(xy0[sides["right"], 0].mean() - xy0[sides["left"], 0].mean())
        height0 = float(xy0[sides["top"], 1].mean() - xy0[sides["bottom"], 1].mean())

        def wh(graph):
            xy = study.physical(graph, reference).x[:, :2].numpy()
            return (float(xy[sides["right"], 0].mean() - xy[sides["left"], 0].mean()),
                    float(xy[sides["top"], 1].mean() - xy[sides["bottom"], 1].mean()))

        truth, reconstruction = [], []
        for frame in range(200):
            latent = encode_frame_latent(
                result["ae"], simulation, frame, pos_dim=2,
                node_feature_mode=result["params"]["node_feature_mode"],
                normalizers=result["normalizers"], device="cpu",
            )
            graph = decode_latent_to_graph(
                result["ae"], simulation, latent, frame, pos_dim=2,
                ae_target_mode=result["params"]["ae_target_mode"],
                normalizers=result["normalizers"], device="cpu",
            )
            truth.append((width0, height0) if frame == 0 else wh(simulation[frame]))
            reconstruction.append(wh(graph))
            latent_rows.append({"cohort": cohort, "source": source, "trajectory_id": trajectory_id,
                                "frame": frame, "z0": float(latent[0]), "z1": float(latent[1])})
        true_width, true_height = np.asarray(truth).T
        pred_width, pred_height = np.asarray(reconstruction).T
        rows.append({
            "cohort": cohort, "source": source, "trajectory_id": trajectory_id,
            "fit_true_p": fit_p(true_width, true_height, width0, height0),
            "fit_pred_p": fit_p(pred_width, pred_height, width0, height0),
            "endpoint_true_p": endpoint_p(true_width[-1], true_height[-1], width0, height0),
            "endpoint_pred_p": endpoint_p(pred_width[-1], pred_height[-1], width0, height0),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / f"{cohort}_per_network.csv", index=False)
    pd.DataFrame(latent_rows).to_csv(output / f"{cohort}_latents.csv", index=False)
    summaries = []
    for source, values in table.groupby("source"):
        summaries.append({
            "cohort": cohort, "source": source, "total": len(values),
            "fit_p_ratio_r2": r2(values.fit_true_p, values.fit_pred_p),
            "fit_p_ratio_mae": float(np.abs(values.fit_true_p - values.fit_pred_p).mean()),
            "endpoint_p_ratio_r2": r2(values.endpoint_true_p, values.endpoint_pred_p),
            "endpoint_p_ratio_mae": float(np.abs(values.endpoint_true_p - values.endpoint_pred_p).mean()),
            "valid/total": f"{len(values)}/{len(values)}",
        })
    pd.DataFrame(summaries).to_csv(output / f"{cohort}_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if args.seed not in protocol["seeds"]:
        raise ValueError(args.seed)
    train_ids = protocol["train_ids"][:2] if args.smoke else protocol["train_ids"]
    validation_ids = protocol["validation_ids"][:2] if args.smoke else protocol["validation_ids"]
    name = f"seed{args.seed}" + ("_smoke" if args.smoke else "")
    output = OUT / name
    output.mkdir(parents=True, exist_ok=False)
    cfg = study.ae_config(output)
    cfg["model_seed"] = args.seed
    source = training_spec(train_ids, validation_ids)
    seed_everything(args.seed)
    result = run_latent_experiment(source, cfg, device="cpu")
    observed_train = {float(sim[0].temperature) for sim in result["train_data"]}
    observed_validation = {float(sim[0].temperature) for sim in result["val_data"]}
    if not observed_train <= {20., 30.} or not observed_validation <= {20., 30.}:
        raise AssertionError((observed_train, observed_validation))
    result["ae_history"].to_csv(output / "ae_warmup_history.csv", index=False)
    ae_sha = base_train.fit_uniform(result, output, 1 if args.smoke else 60)
    if args.smoke:
        evaluate(result, protocol["matched_low_temperature_ids"][:1], "matched_low", output)
        evaluate(result, protocol["unseen_low_temperature_ids"][:1], "unseen_low", output)
    else:
        evaluate(result, protocol["matched_low_temperature_ids"], "matched_low", output)
        evaluate(result, protocol["unseen_low_temperature_ids"], "unseen_low", output)
    metadata = {
        "status": "complete", "seed": args.seed, "smoke": args.smoke,
        "train_ids": train_ids, "validation_ids": validation_ids,
        "train_temperatures_K": sorted(observed_train),
        "validation_temperatures_K": sorted(observed_validation),
        "ae_sha256": ae_sha, "protocol_sha256": sha256(HERE / "protocol.json"),
        "script_sha256": sha256(Path(__file__)), "final_test_used": False,
        "job_id": os.environ.get("PBS_JOBID"),
    }
    (output / "completed.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    main()
