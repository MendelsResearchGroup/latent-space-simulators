#!/usr/bin/env python3
"""Collect five-seed p-ratio shards and report source-wise accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CADENCES = ("1000md", "200md")
BRANCHES = ("node", "stiffness")
PROFILES = ("low", "mixed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(group: pd.DataFrame, truth: str, prediction: str) -> dict[str, object]:
    pair = group[[truth, prediction]].replace([np.inf, -np.inf], np.nan).dropna()
    total = len(group)
    if len(pair) < 2:
        return {"valid": len(pair), "total": total}
    y = pair[truth].to_numpy(float)
    yhat = pair[prediction].to_numpy(float)
    residual = yhat - y
    denominator = float(np.square(y - y.mean()).sum())
    correlation = float(np.corrcoef(y, yhat)[0, 1])
    return {
        "valid": int(len(pair)),
        "total": int(total),
        "r2": float(1.0 - np.square(residual).sum() / denominator) if denominator > 0 else None,
        "pearson_r": correlation if np.isfinite(correlation) else None,
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "mean_signed_error": float(residual.mean()),
    }


def collect(root: Path) -> None:
    windows = []
    prefixes = []
    input_hashes = {}
    for cadence in CADENCES:
        for branch in BRANCHES:
            for profile in PROFILES:
                shard = root / "shards" / f"{cadence}_{branch}_{profile}"
                metadata = json.loads((shard / "metadata.json").read_text())
                if metadata.get("status") != "complete":
                    raise RuntimeError(f"Incomplete shard: {shard}")
                if (metadata["cadence"], metadata["branch"], metadata["profile"]) != (
                    cadence,
                    branch,
                    profile,
                ):
                    raise RuntimeError(f"Shard identity mismatch: {shard}")
                for filename, target in (
                    ("per_network_window.csv", windows),
                    ("per_network_prefix.csv", prefixes),
                ):
                    path = shard / filename
                    frame = pd.read_csv(path)
                    if frame.source_index.nunique() != 200:
                        raise RuntimeError(f"Expected 200 networks: {path}")
                    target.append(frame)
                    input_hashes[str(path)] = sha256(path)

    window = pd.concat(windows, ignore_index=True)
    prefix = pd.concat(prefixes, ignore_index=True)
    window.to_csv(root / "per_network_window.csv", index=False)
    prefix.to_csv(root / "per_network_prefix.csv", index=False)

    summary_rows = []
    estimators = (
        "p_ratio",
        "replicate_mean_p_ratio",
        "ensemble_mean_p_ratio",
        "endpoint_ensemble_p_ratio",
    )
    truths = ("recorded_ground_truth_p_ratio", "source_rollout_p_ratio")
    for keys, group in window.groupby(["cadence", "branch", "profile", "window"], sort=False):
        for truth in truths:
            for estimator in estimators:
                summary_rows.append(
                    {
                        **dict(zip(("cadence", "branch", "profile", "window"), keys)),
                        "temperature_K": "all",
                        "truth": truth,
                        "estimator": estimator,
                        **metrics(group, truth, estimator),
                    }
                )
                if group.profile.iloc[0] == "mixed":
                    for temperature, subset in group.groupby("temperature_K"):
                        summary_rows.append(
                            {
                                **dict(zip(("cadence", "branch", "profile", "window"), keys)),
                                "temperature_K": temperature,
                                "truth": truth,
                                "estimator": estimator,
                                **metrics(subset, truth, estimator),
                            }
                        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "summary_by_group.csv", index=False)

    prefix_rows = []
    for keys, group in prefix.groupby(
        ["cadence", "branch", "profile", "stop_frame", "stop_md_step"], sort=False
    ):
        for estimator in estimators:
            prefix_rows.append(
                {
                    **dict(
                        zip(
                            ("cadence", "branch", "profile", "stop_frame", "stop_md_step"),
                            keys,
                        )
                    ),
                    "estimator": estimator,
                    **metrics(group, "recorded_ground_truth_p_ratio", estimator),
                }
            )
    pd.DataFrame(prefix_rows).to_csv(root / "summary_by_horizon.csv", index=False)

    matched = window[
        ((window.cadence == "1000md") & (window.window == "standard"))
        | ((window.cadence == "200md") & (window.window == "matched_199k"))
    ]
    pivot = matched.pivot_table(
        index=["branch", "profile", "source_index", "source_chunk", "temperature_K"],
        columns="cadence",
        values="p_ratio",
    ).reset_index()
    pivot["difference_200md_minus_1000md"] = pivot["200md"] - pivot["1000md"]
    pivot.to_csv(root / "matched_horizon_cadence_comparison.csv", index=False)

    metadata = {
        "status": "complete",
        "shards": 8,
        "network_window_rows": len(window),
        "network_prefix_rows": len(prefix),
        "input_sha256": input_hashes,
        "script_sha256": sha256(Path(__file__)),
        "primary_estimator": "p_ratio (five-replica fixed-effects shared slope)",
        "comparators": list(estimators[1:]),
        "matched_cadence_horizon_md_steps": [30_000, 199_000],
        "final_test_used": False,
    }
    (root / "completed.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root)


if __name__ == "__main__":
    main()
