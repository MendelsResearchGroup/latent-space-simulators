#!/usr/bin/env python3
"""Freeze and submit the 24-cell nested latent-count matrix without duplicate fits."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
HERE = ROOT / "hpc/experiments/mini_temperature200_frames500"
OUT = ROOT / "notebooks/results/mini_temperature200_frames500"
CODE = OUT / "code_v3_morph"
LEDGER = OUT / "job_ledger.json"
DATA_JOB = "4687263.zeus-master"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted(command: list[str]) -> str:
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+(?:\[\])?\.[\w.-]+", job), job
    return job


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    assert not CODE.exists()
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("morph.py", "morph.pbs", "morph_array.pbs", "morph_matrix.json", "evaluate.py", "protocol.json", "submit_morph.py"):
        shutil.copy2(HERE / name, CODE / name)
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    jobs = {}
    jobs["fresh_22_cell_array"] = accepted(["qsub", "-J", "0-21", "-W", f"depend=afterok:{DATA_JOB}", "-v", f"LSS_CODE_ROOT={CODE}", str(CODE / "morph_array.pbs")])
    reuse_dependencies = {1: "4687264.zeus-master", 20: "4687265.zeus-master"}
    for count, dependency in reuse_dependencies.items():
        jobs[f"reuse_count{count}_seed123"] = accepted(["qsub", "-W", f"depend=afterok:{dependency}", "-v", f"LSS_CODE_ROOT={CODE},LSS_COUNT={count},LSS_SEED=123,LSS_REUSE_FLAG=--reuse", str(CODE / "morph.pbs")])
    record = {
        "status": "submitted", "submitted_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(CODE.relative_to(ROOT)), "code_manifest_sha256": sha(CODE / "manifest.json"),
        "matrix_sha256": sha(CODE / "morph_matrix.json"), "protocol_sha256": sha(CODE / "protocol.json"),
        "jobs": jobs, "cells": 24, "fresh_fits": 22,
        "reuse": "count1/seed123 and count20/seed123 reuse the matching base all4 checkpoints after their fit jobs complete",
        "outputs": "all80 heldout initial states plus complete500-frame paths for fixed IDs40..59 for each low-T source/cell",
        "resources_each": {"queue": "mendels_q", "placement": "place=free:shared", "ncpus": 8, "memory": "16gb", "walltime": "24:00:00"},
    }
    ledger["latent_count_matrix"] = record
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
