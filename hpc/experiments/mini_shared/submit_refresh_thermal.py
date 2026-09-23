#!/usr/bin/env python3
"""Freeze and submit the five mixed-only thermal p-ratio refresh jobs."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "hpc/experiments/mini_shared"
OUT = ROOT / "notebooks/results/mini_shared"
THERMAL = OUT / "thermal_fit"
CODE = THERMAL / "code_thermal_fit_v1"
LEDGER = THERMAL / "job_ledger.json"
ARMS = ("single_stiffness", "single_node", "stiffness_only", "joint_original", "joint_all")
THERMAL_METRIC_SOURCE = ROOT / "hpc/experiments/ml_only_engineering/thermal_metric.py"
GRAPH_UTILS_MAIN_COMMIT = "734c6b8"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(arm: str) -> str:
    variables = f"LSS_CODE_ROOT={CODE},LSS_ARM={arm}"
    command = ["qsub", "-N", "mini_thermal_fit", "-v", variables, str(CODE / "refresh_thermal.pbs")]
    job = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r"\d+\.[\w.-]+", job):
        raise ValueError(f"Unexpected qsub output: {job!r}")
    return job


def main() -> None:
    if CODE.exists() or LEDGER.exists():
        raise FileExistsError("thermal refresh snapshot or ledger already exists; preserving a previous submission.")
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Data manifest is incomplete.")
    for arm in ARMS:
        source = OUT / arm
        if json.loads((source / "fit_completed.json").read_text()).get("status") != "completed":
            raise ValueError(f"Frozen fit incomplete: {arm}")
        for checkpoint in (source / "ae.pt", source / "propagator.pt"):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

    CODE.mkdir(parents=True)
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("refresh_thermal.py", "refresh_thermal.pbs", "submit_refresh_thermal.py", "collect.py", "evaluate.py", "thermal_response.py", "train.py"):
        shutil.copyfile(HERE / name, CODE / name)
    shutil.copyfile(THERMAL_METRIC_SOURCE, CODE / "thermal_metric.py")
    shutil.copyfile(OUT / "data_manifest.json", CODE / "data_manifest.json")
    files = {str(path.relative_to(CODE)): sha(path) for path in sorted(CODE.rglob("*")) if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")

    frozen_models = {
        arm: {"ae_sha256": sha(OUT / arm / "ae.pt"), "propagator_sha256": sha(OUT / arm / "propagator.pt")}
        for arm in ARMS
    }
    ledger = {
        "study": "mini_shared",
        "operation": "mixed-only thermal p-ratio post-fit refresh; no training",
        "snapshot": str(CODE.relative_to(ROOT)),
        "code_manifest_sha256": sha(CODE / "manifest.json"),
        "data_manifest_sha256": sha(CODE / "data_manifest.json"),
        "legacy_thermal_metric": {
            "source": str(THERMAL_METRIC_SOURCE.relative_to(ROOT)),
            "source_sha256": sha(THERMAL_METRIC_SOURCE),
            "snapshot_sha256": sha(CODE / "thermal_metric.py"),
        },
        "graph_utils_main_commit_checked": GRAPH_UTILS_MAIN_COMMIT,
        "input_data_sha256": {name: {"trainval": entry["trainval_sha256"], "test": entry["test_sha256"]} for name, entry in manifest["datasets"].items()},
        "frozen_model_sha256_before": frozen_models,
        "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
        "jobs": {},
        "status": "submitting",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    for arm in ARMS:
        ledger["jobs"][arm] = qsub(arm)
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger["status"] = "submitted"
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
