#!/usr/bin/env python3
"""Freeze code and submit all six matched AE/propagator arms behind data audit."""
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
CODE = OUT / "code_v1"
LEDGER = OUT / "job_ledger.json"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(script: Path, arm: str, dependency: str) -> str:
    command = ["qsub", "-N", f"m2f5_{arm[:7]}", "-W", f"depend=afterok:{dependency}", "-v", f"LSS_CODE_ROOT={CODE},LSS_ARM={arm}", str(script)]
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job), job
    return job


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    assert ledger["status"] == "preparing_data"
    assert not CODE.exists()
    environment = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mt200-submit", PYTHONPATH=f"{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src")
    subprocess.run([PYTHON, str(HERE / "train.py"), "--assert-config"], check=True, env=environment)
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "evaluate.py", "collect.py", "fit.pbs", "collect.pbs", "protocol.json", "prepare.py", "prepare.pbs", "submit.py"):
        shutil.copy2(HERE / name, CODE / name)
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    protocol = json.loads((HERE / "protocol.json").read_text())
    record = {
        "snapshot": str(CODE.relative_to(ROOT)), "code_manifest_sha256": sha(CODE / "manifest.json"),
        "protocol_sha256": sha(CODE / "protocol.json"), "submitted_at": datetime.now(timezone.utc).isoformat(),
        "data_dependency": ledger["data_preparation_job"],
        "resources_each": {"queue": "mendels_q", "placement": "place=free:shared", "ncpus": 8, "memory": "16gb", "walltime": "24:00:00"},
        "jobs": {},
    }
    ledger["status"] = "base_matrix_submitted_waiting_on_data"
    ledger["base_matrix"] = record
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    for arm in protocol["base_arms"]:
        fit = qsub(CODE / "fit.pbs", arm, ledger["data_preparation_job"])
        record["jobs"][arm] = {"fit": fit}
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    for arm in protocol["base_arms"]:
        collect = qsub(CODE / "collect.pbs", arm, record["jobs"][arm]["fit"])
        record["jobs"][arm]["collector"] = collect
        LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
