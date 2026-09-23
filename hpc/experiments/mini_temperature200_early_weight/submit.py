#!/usr/bin/env python3
"""Freeze and submit the matched early-weighted pilot behind an end-to-end smoke."""
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
HERE = ROOT / "hpc/experiments/mini_temperature200_early_weight"
SOURCE = ROOT / "hpc/experiments/mini_temperature200_frames500"
SOURCE_RESULTS = ROOT / "notebooks/results/mini_temperature200_frames500"
OUT = ROOT / "notebooks/results/mini_temperature200_early_weight"
CODE = OUT / "code_v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submit(script: str, dependency: str | None = None) -> str:
    command = ["qsub"]
    if dependency is not None:
        command.extend(["-W", f"depend=afterok:{dependency}"])
    command.extend(["-v", f"LSS_CODE_ROOT={CODE}", str(CODE / script)])
    job = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r"\d+\.[\w.-]+", job):
        raise RuntimeError(f"Unexpected qsub response: {job!r}")
    return job


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(HERE / "train.py", CODE / "train.py")
    shutil.copy2(SOURCE / "train.py", CODE / "base_train.py")
    shutil.copy2(SOURCE / "collect.py", CODE / "collect.py")
    shutil.copy2(SOURCE / "evaluate.py", CODE / "evaluate.py")
    for name in ("protocol.json", "smoke.pbs", "fit.pbs", "collect.pbs"):
        shutil.copy2(HERE / name, CODE / name)
    shutil.copy2(SOURCE_RESULTS / "data_manifest.json", OUT / "data_manifest.json")
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    ledger = {
        "study": "mini_temperature200_early_weight",
        "status": "submitting",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(CODE.relative_to(ROOT)),
        "code_manifest_sha256": sha(CODE / "manifest.json"),
        "protocol_sha256": sha(CODE / "protocol.json"),
        "data_manifest_sha256": sha(OUT / "data_manifest.json"),
        "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
        "jobs": {},
    }
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    smoke = submit("smoke.pbs")
    ledger["jobs"]["smoke"] = smoke
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    fit = submit("fit.pbs", smoke)
    ledger["jobs"]["fit"] = fit
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    collect = submit("collect.pbs", fit)
    ledger["jobs"]["collect"] = collect
    ledger["status"] = "submitted"
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
