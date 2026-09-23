#!/usr/bin/env python3
"""Freeze the mini-shared recipe and submit its smoke-gated fitting/evaluation jobs."""
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
CODE = OUT / "code_v1"
ARMS = ("single_stiffness", "single_node", "stiffness_only", "joint_original")
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
    if manifest.get("status") != "complete" or manifest.get("pbs_job_id") != DATA_AUDIT_JOB:
        raise ValueError("Refusing submission without the successful final data-audit manifest.")
    required = {"node_original", "stiffness_original", "node_mixed", "stiffness_mixed"}
    if set(manifest.get("datasets", ())) != required:
        raise ValueError("Data manifest does not contain the final four-dataset contract.")
    if CODE.exists() or (OUT / "job_ledger.json").exists():
        raise FileExistsError("mini_shared snapshot or ledger already exists; preserving prior submission.")

    CODE.mkdir(parents=True)
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "collect.py", "evaluate.py", "pipeline.pbs", "collect.pbs", "protocol.json", "submit.py"):
        shutil.copyfile(HERE / name, CODE / name)
    shutil.copyfile(OUT / "data_manifest.json", CODE / "data_manifest.json")
    files = {
        str(path.relative_to(CODE)): sha(path)
        for path in sorted(CODE.rglob("*")) if path.is_file()
    }
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    code_manifest_sha256 = sha(CODE / "manifest.json")

    ledger = {
        "study": "mini_shared",
        "snapshot": str(CODE.relative_to(ROOT)),
        "code_manifest_sha256": code_manifest_sha256,
        "protocol_sha256": sha(CODE / "protocol.json"),
        "data_manifest_sha256": sha(CODE / "data_manifest.json"),
        "data_audit_job": DATA_AUDIT_JOB,
        "data_audit_dependency": f"afterok:{DATA_AUDIT_JOB}",
        "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
        "arms": {},
        "collectors": {},
        "status": "submitting",
        "identity_mapping": "Explicit val split order is preserved by resolve_dataset_splits; cached test local indices 0..19/0..79 map to original identities 40..59/40..119.",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for arm in ARMS:
        ledger["arms"][arm] = qsub(
            CODE / "pipeline.pbs",
            {"LSS_CODE_ROOT": str(CODE), "LSS_ARM": arm},
            f"afterok:{DATA_AUDIT_JOB}",
            "mini_shared",
        )
        (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    fit_dependency = "afterok:" + ":".join(ledger["arms"][arm] for arm in ARMS)
    for arm in ARMS:
        ledger["collectors"][arm] = qsub(
            CODE / "collect.pbs",
            {"LSS_CODE_ROOT": str(CODE), "LSS_ARM": arm},
            fit_dependency,
            "mini_shared_eval",
        )
        (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    ledger["fit_dependency"] = fit_dependency
    ledger["status"] = "submitted"
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger, indent=2), flush=True)


if __name__ == "__main__":
    main()
