#!/usr/bin/env python3
"""Export heldout mixed-T latent paths from an existing frozen checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import encode_frame_latent


ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
BASE = ROOT / "notebooks/results/mini_temperature200_frames500"
SOURCES = ("node_mixed", "stiffness_mixed")
FRAMES = 500


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    folder = BASE / args.folder
    recipe = json.loads((folder / "propagator_recipe.json").read_text())
    result = load_experiment_bundle(folder / "propagator.pt", cfg=recipe["config"], device="cpu")
    model, params, normalizers = result["ae"], result["params"], result["normalizers"]
    model.eval()
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    rows = []
    with torch.no_grad():
        for name in SOURCES:
            entry = manifest["datasets"][name]
            assert sha(Path(entry["cache_path"])) == entry["cache_sha256"]
            ids = list(range(40, 120))
            item = {"name": name, "label": name, "path": entry["cache_path"], "train_count": 0,
                    "val_count": len(ids), "split_indices": {"train": [], "val": ids, "test": []}}
            source = {"name": name, "label": name, "path": entry["cache_path"], "dataset_name": name,
                      "source_name": name, "coordinate_normalization": "position_normalization",
                      "dataset_mixture": [item]}
            _, simulations, _, _ = resolve_train_val_test(source, params, split_seed=123)
            assert [int(sim[0].source_sim_id) for sim in simulations] == ids
            for simulation in simulations:
                reference = simulation[0]
                for frame in range(FRAMES):
                    z = encode_frame_latent(model, simulation, frame, pos_dim=2,
                        node_feature_mode=params["node_feature_mode"], normalizers=normalizers,
                        device="cpu").detach().cpu().reshape(-1)
                    rows.append({"source": name, "trajectory_id": int(reference.source_sim_id),
                                 "frame": frame, "md_step": frame * 200,
                                 "z0": float(z[0]), "z1": float(z[1])})
    output = folder / "test/mixed_temperature_latents.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    receipt = {"status": "completed", "job_id": os.environ.get("PBS_JOBID"),
               "folder": args.folder, "sources": list(SOURCES), "heldout_ids": [40, 119],
               "frames": [0, 499], "rows": len(rows), "checkpoint_sha256": sha(folder / "ae.pt"),
               "output_sha256": sha(output)}
    (folder / "test/mixed_temperature_latents_completed.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
