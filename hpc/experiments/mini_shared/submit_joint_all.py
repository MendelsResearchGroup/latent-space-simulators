#!/usr/bin/env python3
"""Freeze and submit the user-authorized all-four-source mini-shared arm."""
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
CODE = OUT / "code_v2"
LEDGER = OUT / "joint_all_job_ledger.json"
DATA_AUDIT_JOB = "4682393.zeus-master"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(script: Path, variables: dict[str, str], dependency: str, name: str) -> str:
    encoded = ",".join(f"{key}={value}" for key, value in variables.items())
    command = ["qsub", "-N", name, "-W", f"depend={dependency}", "-v", encoded, str(script)]
    job = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r"\d+\.[\w.-]+", job):
        raise ValueError(f"Unexpected qsub output: {job!r}")
    return job


def main() -> None:
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    expected = ["stiffness_original", "node_original", "stiffness_mixed", "node_mixed"]
    if manifest.get("status") != "complete" or manifest.get("pbs_job_id") != DATA_AUDIT_JOB:
        raise ValueError("Refusing submission without the successful final data-audit manifest.")
    if protocol.get("arms", {}).get("joint_all") != expected:
        raise ValueError("Live protocol does not contain the authorized joint_all source order.")
    if CODE.exists() or LEDGER.exists():
        raise FileExistsError("joint_all snapshot or ledger already exists; preserving prior submission.")

    CODE.mkdir(parents=True)
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "collect.py", "evaluate.py", "pipeline.pbs", "collect.pbs", "protocol.json", "submit.py", "submit_joint_all.py"):
        shutil.copyfile(HERE / name, CODE / name)
    shutil.copyfile(OUT / "data_manifest.json", CODE / "data_manifest.json")
    files = {str(path.relative_to(CODE)): sha(path) for path in sorted(CODE.rglob("*")) if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")

    ledger = {
        "study": "mini_shared",
        "arm": "joint_all",
        "source_order": expected,
        "snapshot": str(CODE.relative_to(ROOT)),
        "code_manifest_sha256": sha(CODE / "manifest.json"),
        "protocol_sha256": sha(CODE / "protocol.json"),
        "data_manifest_sha256": sha(CODE / "data_manifest.json"),
        "data_audit_dependency": f"afterok:{DATA_AUDIT_JOB}",
        "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
        "status": "submitting",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    ledger["fit_job"] = qsub(CODE / "pipeline.pbs", {"LSS_CODE_ROOT": str(CODE), "LSS_ARM": "joint_all"}, f"afterok:{DATA_AUDIT_JOB}", "mini_shared")
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    ledger["collector_job"] = qsub(CODE / "collect.pbs", {"LSS_CODE_ROOT": str(CODE), "LSS_ARM": "joint_all"}, f"afterok:{ledger['fit_job']}", "mini_shared_eval")
    ledger["status"] = "submitted"
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2), flush=True)


if __name__ == "__main__":
    main()
