#!/usr/bin/env python3
"""Freeze the loader-only repair and submit a fresh gated chain."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
HERE = ROOT / "hpc/experiments/mixed_temperature_extrapolation_2frame"
OUT = ROOT / "notebooks/results/mixed_temperature_extrapolation_2frame"
CODE = OUT / "code_v2"
BASE_CODE = ROOT / "notebooks/results/all4_coverage_1000/code_v5"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submit(script: str, dependency: str | None = None, tag: str = "") -> str:
    variables = f"LSS_CODE_ROOT={CODE}"
    if tag:
        variables += f",LSS_RUN_TAG={tag}"
    command = ["qsub"]
    if dependency:
        command += ["-W", "depend=afterok:" + dependency]
    command += ["-v", variables, str(CODE / script)]
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job)
    return job


def main() -> None:
    if CODE.exists():
        raise FileExistsError(CODE)
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run.py", "protocol.json", "smoke.pbs", "ae.pbs", "dynamics.pbs", "submit.py", "retry_loader.py"):
        shutil.copy2(HERE / name, CODE / name)
    shutil.copy2(BASE_CODE / "train.py", CODE / "base_train.py")
    shutil.copy2(BASE_CODE / "evaluate.py", CODE / "evaluate.py")
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")

    ledger_path = OUT / "job_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["status"] = "retry1_submitting"
    ledger["failure"] = {
        "job": ledger["jobs"]["smoke"],
        "stage": "dynamics checkpoint load",
        "exit_status": 1,
        "reason": "generic capacity loader expected a spec key absent from the standard ae_only checkpoint",
        "scientific_result": False,
    }
    ledger["retry1"] = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "code_snapshot": str(CODE.relative_to(ROOT)),
        "code_manifest_sha256": sha(CODE / "manifest.json"),
        "change": "loader-only: use the matching standard ae_only cache loader; add distinct retry smoke output names",
        "jobs": {},
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    smoke = submit("smoke.pbs", tag="_retry1")
    ledger["retry1"]["jobs"]["smoke"] = smoke
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ae = submit("ae.pbs", dependency=smoke)
    ledger["retry1"]["jobs"]["ae"] = ae
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    dynamics = submit("dynamics.pbs", dependency=ae)
    ledger["retry1"]["jobs"]["dynamics"] = dynamics
    ledger["status"] = "retry1_submitted"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger["retry1"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
