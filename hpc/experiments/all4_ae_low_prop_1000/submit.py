#!/usr/bin/env python3
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
HERE = ROOT / "hpc/experiments/all4_ae_low_prop_1000"
OUT = ROOT / "notebooks/results/all4_ae_low_prop_1000"
CODE = OUT / "code_v1"
SOURCE = ROOT / "notebooks/results/all4_coverage_1000/all4_twenty_each_retry_v5_validation30"
DATA = ROOT / "notebooks/results/all4_coverage_1000/data_manifest.json"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(script: str, dependency: str | None = None) -> str:
    command = ["qsub"]
    if dependency:
        command += ["-W", "depend=afterok:" + dependency]
    command += ["-v", f"LSS_CODE_ROOT={CODE}", str(CODE / script)]
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job)
    return job


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    assert SOURCE.joinpath("ae.pt").is_file() and SOURCE.joinpath("propagator_recipe.json").is_file()
    OUT.mkdir(parents=True)
    shutil.copy2(DATA, OUT / "data_manifest.json")
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "collect.py", "linear_fluctuation.py", "protocol.json", "pipeline.pbs", "collect.pbs", "submit.py"):
        shutil.copy2(HERE / name, CODE / name)
    shutil.copy2(ROOT / "notebooks/results/all4_coverage_1000/code_v5/evaluate.py", CODE / "evaluate.py")
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    environment = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), LSS_CODE_ROOT=str(CODE), OMP_NUM_THREADS="1",
                       PYTHONPATH=f"{CODE / 'src'}:{CODE}:/rg/mendels_prj/alexander.z/MetaForge/src")
    subprocess.run([PYTHON, str(CODE / "train.py"), "--assert-config"], check=True, env=environment)
    ledger = {
        "study": "all4_ae_low_prop_1000", "status": "submitting",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(CODE.relative_to(ROOT)), "code_manifest_sha256": sha(CODE / "manifest.json"),
        "data_manifest_sha256": sha(OUT / "data_manifest.json"),
        "source_ae": str(SOURCE.joinpath("ae.pt").relative_to(ROOT)), "source_ae_sha256": sha(SOURCE / "ae.pt"),
        "split": "propagator low-T train IDs0..19, validation IDs20..39; all-four heldout-development IDs40..119",
        "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
        "jobs": {},
    }
    ledger_path = OUT / "job_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    fit = qsub("pipeline.pbs")
    ledger["jobs"]["fit"] = fit
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    collector = qsub("collect.pbs", fit)
    ledger["jobs"]["collector"] = collector
    ledger["status"] = "submitted"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
