"""Cumulative mixed-T affine-strain p-ratio versus horizon for notebook 02."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PREFIX_CODE = ROOT / "hpc/experiments/mini_temperature1000_linear_prefix"
ALL4_CODE = ROOT / "notebooks/results/all4_coverage_1000/code_v5"
for path in (PREFIX_CODE, ALL4_CODE, ROOT / "src", ROOT.parent / "MetaForge/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
SPEC = importlib.util.spec_from_file_location("frozen_prefix", PREFIX_CODE / "run.py")
FROZEN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FROZEN)

BASE = ROOT / "notebooks/results/all4_coverage_1000"
OUT = BASE / "prefix_linear_fit"
ARMS = {
    "all4_one_each_retry_v5_validation30": ("node_mixed", "stiffness_mixed"),
    "all4_twenty_each_retry_v5_validation30": ("node_mixed", "stiffness_mixed"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    for arm, sources in ARMS.items():
        folder = OUT / arm
        folder.mkdir(exist_ok=False)
        model = BASE / arm
        before = {"ae": sha(model / "ae.pt"), "propagator": sha(model / "propagator.pt")}
        recipe = json.loads((model / "propagator_recipe.json").read_text())
        result = FROZEN.load_experiment_bundle(model / "propagator.pt", cfg=recipe["config"], device="cpu")
        rows = []
        for source in sources:
            data = FROZEN.load_sims(result, manifest["datasets"][source], source)
            rows += FROZEN.evaluate_source(result, data, source, arm)
        table = pd.DataFrame(rows)
        assert table.groupby(["source", "mode", "trajectory_id"]).horizon.nunique().eq(len(FROZEN.HORIZONS)).all()
        table.to_csv(folder / "per_network_prefix.csv", index=False)
        summaries = []
        for (source, mode, horizon), group in table.groupby(["source", "mode", "horizon"]):
            good = group.dropna(subset=["linear_true_p", "linear_pred_p"])
            summaries.append({"source": source, "mode": mode, "horizon": horizon, "md_step": horizon * 1000, "linear_p_r2": FROZEN.r2(good.linear_true_p, good.linear_pred_p), "linear_p_mae": float(np.abs(good.linear_true_p - good.linear_pred_p).mean()), "linear_valid": len(good), "linear_total": len(group), "true_fit_r2_median": float(good.linear_true_fit_r2.median()), "pred_fit_r2_median": float(good.linear_pred_fit_r2.median())})
        pd.DataFrame(summaries).to_csv(folder / "summary_prefix.csv", index=False)
        parity = []
        final = table[table.horizon.eq(199)]
        old = pd.read_csv(model / "auxiliary_linear_fit/per_network.csv")
        for (source, mode), group in final.groupby(["source", "mode"]):
            expected = old[(old.source == source) & (old["mode"] == mode)][["trajectory_id", "linear_true_p", "linear_pred_p"]]
            joined = group[["trajectory_id", "linear_true_p", "linear_pred_p"]].merge(expected, on="trajectory_id", suffixes=("_prefix", "_final"), validate="one_to_one")
            errors = {"true": float(np.abs(joined.linear_true_p_prefix - joined.linear_true_p_final).max()), "pred": float(np.abs(joined.linear_pred_p_prefix - joined.linear_pred_p_final).max())}
            assert len(joined) == 80 and max(errors.values()) < 2e-8
            parity.append({"source": source, "mode": mode, "rows": len(joined), **errors})
        after = {"ae": sha(model / "ae.pt"), "propagator": sha(model / "propagator.pt")}
        assert before == after
        (folder / "completed.json").write_text(json.dumps({"status": "completed", "arm": arm, "job_id": os.environ.get("PBS_JOBID"), "checkpoint_sha256_before": before, "checkpoint_sha256_after": after, "horizons": [37, 199], "fit_window": "frames30..horizon inclusive", "parity_at_frame199": parity, "postfit_only": True, "script_sha256": sha(Path(__file__))}, indent=2, sort_keys=True) + "\n")
        del result
    print("completed cumulative mixed-T prefix evaluation", flush=True)


if __name__ == "__main__":
    main()
