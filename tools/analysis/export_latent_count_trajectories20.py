"""Export complete paths for 20 matched held-out stiffness graphs per count."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
os.environ.setdefault("LSS_PROJECT_ROOT", str(ROOT))
for source_root in (ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from lss.dynamics.experiment import _load_ae_cache, resolve_train_val_test
from lss.dynamics.simulation import batch_delta_graphs


BASE = ROOT / "notebooks/results/all4_coverage_1000"
MORPH = ROOT / "notebooks/results/latent_count_morph"
OUT = MORPH / "collected"
COUNTS = (1, 2, 4, 8, 10, 12, 14, 20)
HELDOUT_IDS = tuple(range(40, 60))
SOURCE = "stiffness_low"
SEED = 123


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_train_module():
    path = BASE / "code_v5/train.py"
    spec = importlib.util.spec_from_file_location("all4_frozen_train_paths20", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def checkpoint_and_config(count: int, frozen_train):
    if count in (1, 20):
        donor = BASE / (
            "all4_one_each_retry_v5_validation30"
            if count == 1
            else "all4_twenty_each_retry_v5_validation30"
        )
        recipe = json.loads((donor / "propagator_recipe.json").read_text())
        return donor / "ae.pt", recipe["config"]
    cell = MORPH / f"c{count}_s{SEED}"
    return cell / "ae.pt", frozen_train.config(cell)


def heldout_stiffness(result: dict, manifest: dict):
    entry = manifest["datasets"][SOURCE]
    spec = {
        "name": SOURCE,
        "label": SOURCE,
        "path": entry["test_path"],
        "train_count": 0,
        "val_count": 80,
        "split_indices": {"train": [], "val": list(range(40, 120)), "test": []},
    }
    source = {
        "name": SOURCE,
        "label": SOURCE,
        "path": entry["test_path"],
        "dataset_name": SOURCE,
        "source_name": SOURCE,
        "coordinate_normalization": "position_normalization",
        "dataset_mixture": [spec],
    }
    _, simulations, _, _ = resolve_train_val_test(
        source, result["params"], split_seed=SEED
    )
    if len(simulations) != 80:
        raise AssertionError(f"Expected 80 held-out simulations, found {len(simulations)}.")
    return simulations[: len(HELDOUT_IDS)]


def encode_frame_batch(result: dict, simulations, frame: int) -> torch.Tensor:
    params, normalizers = result["params"], result["normalizers"]
    packed = batch_delta_graphs(
        simulations,
        [(index, frame) for index in range(len(simulations))],
        pos_dim=2,
        device="cpu",
        node_feature_mode=params["node_feature_mode"],
        edge_mode=params["edge_mode"],
    )
    node = (
        packed["node_feature"] - normalizers["node_feature_mean"]
    ) / normalizers["node_feature_std"]
    edge = (packed["edge_attr"] - normalizers["edge_mean"]) / normalizers["edge_std"]
    ref_edge = (
        packed["ref_edge_attr"]
        - normalizers.get("ref_edge_mean", normalizers["edge_mean"])
    ) / normalizers.get("ref_edge_std", normalizers["edge_std"])
    latent, _ = result["ae"].encode(
        node,
        packed["ref_pos"],
        edge,
        ref_edge,
        packed["edge_index"],
        packed["batch"],
    )
    return latent.detach().cpu()


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    manifest_path = BASE / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frozen_train = frozen_train_module()
    rows = []
    checkpoints = {}

    for count in COUNTS:
        checkpoint, config = checkpoint_and_config(count, frozen_train)
        completed = json.loads((MORPH / f"c{count}_s{SEED}" / "completed.json").read_text())
        if sha256(checkpoint) != completed["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for count {count}.")
        result = _load_ae_cache(checkpoint, config, device="cpu")
        result["ae"].eval()
        simulations = heldout_stiffness(result, manifest)
        with torch.no_grad():
            for frame in range(200):
                latent = encode_frame_batch(result, simulations, frame)
                for row, trajectory_id in zip(latent.tolist(), HELDOUT_IDS):
                    rows.append({
                        "count": count,
                        "seed": SEED,
                        "source": SOURCE,
                        "trajectory_id": trajectory_id,
                        "frame": frame,
                        "z0": row[0],
                        "z1": row[1],
                    })
        checkpoints[str(count)] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
        }
        print(f"count {count}: encoded {len(simulations)} trajectories x 200 frames", flush=True)
        del simulations, result
        gc.collect()

    table = pd.DataFrame(rows)
    expected = len(COUNTS) * len(HELDOUT_IDS) * 200
    if len(table) != expected:
        raise AssertionError(f"Expected {expected} rows, found {len(table)}.")
    output_path = OUT / "trajectory_latents_20_seed123.csv"
    table.to_csv(output_path, index=False)
    metadata = {
        "status": "complete",
        "seed": SEED,
        "source": SOURCE,
        "counts": list(COUNTS),
        "heldout_ids": list(HELDOUT_IDS),
        "frames": list(range(200)),
        "rows": len(table),
        "checkpoints": checkpoints,
        "data_manifest_sha256": sha256(manifest_path),
        "response_use": "none",
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": sha256(output_path),
    }
    (OUT / "trajectory_latents_20_seed123.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
