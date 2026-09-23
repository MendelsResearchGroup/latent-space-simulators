"""Evaluate the hybrid checkpoint on the frozen held-out development cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from evaluate import evaluate
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from train import ARM, DATA_OUT, OUT, SOURCE_AE, sha, dump


SOURCES = ("node_low", "node_mixed", "stiffness_low", "stiffness_mixed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    folder = OUT / (ARM + ("_smoke" if args.smoke else ""))
    receipt = json.loads((folder / "fit_completed.json").read_text())
    assert receipt["status"] == "completed"
    recipe = json.loads((folder / "recipe.json").read_text())
    result = load_experiment_bundle(folder / "propagator.pt", cfg=recipe["config"], device="cpu")
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    rows, steps = [], []
    cohort = [40] if args.smoke else list(range(40, 120))
    for name in SOURCES:
        entry = manifest["datasets"][name]
        assert sha(Path(entry["test_path"])) == entry["test_sha256"]
        spec = {"name": name, "label": name, "path": entry["test_path"], "train_count": 0,
                "val_count": len(cohort), "split_indices": {"train": [], "val": cohort, "test": []}}
        source = {"name": name, "label": name, "path": entry["test_path"], "dataset_name": name,
                  "source_name": name, "coordinate_normalization": "position_normalization", "dataset_mixture": [spec]}
        _, simulations, _, _ = resolve_train_val_test(source, result["params"], split_seed=123)
        assert len(simulations) == len(cohort)
        for trajectory_id, simulation in zip(cohort, simulations):
            for frame, graph in enumerate(simulation):
                graph.source_sim_id = trajectory_id
                assert graph.time == frame * 1000
        table = evaluate({**result, "val_data": simulations}, seed=123, latent_dim=2,
                         output=folder / "test" / name, horizon=199, split="test")
        rows.append(table)
        steps.append(pd.read_csv(folder / "test" / name / "per_step.csv"))
    pd.concat(rows, ignore_index=True).to_csv(folder / "test/per_network.csv", index=False)
    pd.concat(steps, ignore_index=True).to_csv(folder / "test/per_step.csv", index=False)
    assert sha(SOURCE_AE) == receipt["frozen_ae_sha256_before"]
    dump(folder / "evaluation_completed.json", {
        "status": "completed",
        "job_id": os.environ.get("PBS_JOBID"),
        "smoke": args.smoke,
        "cohort": cohort,
        "frozen_ae_sha256": sha(SOURCE_AE),
        "propagator_sha256": sha(folder / "propagator.pt"),
        "test_used_for_selection": False,
    })


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    main()
