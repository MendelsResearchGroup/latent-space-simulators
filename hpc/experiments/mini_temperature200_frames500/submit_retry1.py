#!/usr/bin/env python3
"""Freeze schema-only retry and resubmit six arms behind corrected data audit."""
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
CODE = OUT / "code_v2"
LEDGER = OUT / "job_ledger.json"
DATA_JOB = "4687263.zeus-master"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(script: Path, arm: str, dependency: str) -> str:
    command = ["qsub", "-N", f"m2f5r1_{arm[:6]}", "-W", f"depend=afterok:{dependency}", "-v", f"LSS_CODE_ROOT={CODE},LSS_ARM={arm}", str(script)]
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job), job
    return job


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    assert not CODE.exists()
    environment = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mt200-retry-submit", PYTHONPATH=f"{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src")
    subprocess.run([PYTHON, str(HERE / "train.py"), "--assert-config"], check=True, env=environment)
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "evaluate.py", "collect.py", "fit.pbs", "collect.pbs", "protocol.json", "prepare.py", "prepare.pbs", "submit.py", "submit_retry1.py"):
        shutil.copy2(HERE / name, CODE / name)
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    record = {
        "reason": "Schema-only data-preparation correction: raw MetaForge graphs derive reference fields in the standard loader.",
        "failed_data_job": ledger["data_preparation_job"], "data_dependency": DATA_JOB,
        "terminalized_fit_jobs": [value["fit"] for value in ledger["base_matrix"]["jobs"].values()],
        "terminalized_collector_jobs": [value["collector"] for value in ledger["base_matrix"]["jobs"].values()],
        "snapshot": str(CODE.relative_to(ROOT)), "code_manifest_sha256": sha(CODE / "manifest.json"),
        "protocol_sha256": sha(CODE / "protocol.json"), "prepare_script_sha256": sha(CODE / "prepare.py"),
        "submitted_at": datetime.now(timezone.utc).isoformat(), "jobs": {},
        "resources_each": {"queue": "mendels_q", "placement": "place=free:shared", "ncpus": 8, "memory": "16gb", "walltime": "24:00:00"},
    }
    ledger["status"] = "retry1_base_matrix_submitted_waiting_on_data"
    ledger["data_preparation_retry1"] = {"job": DATA_JOB, "prepare_script_sha256": record["prepare_script_sha256"]}
    ledger.setdefault("retries", []).append(record)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    protocol = json.loads((CODE / "protocol.json").read_text())
    for arm in protocol["base_arms"]:
        record["jobs"][arm] = {"fit": qsub(CODE / "fit.pbs", arm, DATA_JOB)}
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    for arm in protocol["base_arms"]:
        record["jobs"][arm]["collector"] = qsub(CODE / "collect.pbs", arm, record["jobs"][arm]["fit"])
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
