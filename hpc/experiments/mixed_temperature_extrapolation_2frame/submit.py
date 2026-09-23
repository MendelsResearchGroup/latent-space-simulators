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

import torch


ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
HERE = ROOT / "hpc/experiments/mixed_temperature_extrapolation_2frame"
OUT = ROOT / "notebooks/results/mixed_temperature_extrapolation_2frame"
CODE = OUT / "code_v1"
DATA = ROOT / "notebooks/results/all4_coverage_1000/data_manifest.json"
BASE_CODE = ROOT / "notebooks/results/all4_coverage_1000/code_v5"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submit(script, dependency=None):
    command = ["qsub"]
    if dependency:
        command += ["-W", "depend=afterok:" + dependency]
    command += ["-v", f"LSS_CODE_ROOT={CODE}", str(CODE / script)]
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job)
    return job


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    protocol = json.loads((HERE / "protocol.json").read_text())
    train, validation, evaluation = map(set, (protocol["train_ids"], protocol["validation_ids"], protocol["evaluation_ids"]))
    assert not train & validation and not train & evaluation and not validation & evaluation
    assert train | validation | evaluation == set(range(120))
    data_manifest = json.loads(DATA.read_text()); audit = {}
    for source in protocol["sources"]:
        path = Path(data_manifest["datasets"][source]["trainval_path"])
        assert sha(path) == data_manifest["datasets"][source]["trainval_sha256"]
        simulations = torch.load(path, map_location="cpu", weights_only=False)
        temperatures = {i: float(simulations[i][0].temperature) for i in range(120)}
        assert {temperatures[i] for i in train | validation} <= {0.1, 1.0}
        assert {temperatures[i] for i in evaluation} <= {5.0, 10.0, 20.0, 30.0}
        audit[source] = {"train": {str(t): sum(temperatures[i] == t for i in train) for t in (0.1, 1.0)},
                         "validation": {str(t): sum(temperatures[i] == t for i in validation) for t in (0.1, 1.0)},
                         "evaluation": {str(t): sum(temperatures[i] == t for i in evaluation) for t in (5.0, 10.0, 20.0, 30.0)}}
    OUT.mkdir(parents=True); shutil.copy2(DATA, OUT / "data_manifest.json")
    (OUT / "split_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    CODE.mkdir(); shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run.py", "protocol.json", "smoke.pbs", "ae.pbs", "dynamics.pbs", "submit.py"):
        shutil.copy2(HERE / name, CODE / name)
    shutil.copy2(BASE_CODE / "train.py", CODE / "base_train.py")
    shutil.copy2(BASE_CODE / "evaluate.py", CODE / "evaluate.py")
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    environment = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), LSS_CODE_ROOT=str(CODE), OMP_NUM_THREADS="1",
                       MPLCONFIGDIR="/tmp/temp2f-static", PYTHONPATH=f"{CODE / 'src'}:{CODE}:/rg/mendels_prj/alexander.z/MetaForge/src")
    subprocess.run([PYTHON, str(CODE / "run.py"), "--stage", "assert"], check=True, env=environment)
    ledger = {"study": protocol["study"], "status": "submitting", "submitted_at": datetime.now(timezone.utc).isoformat(),
              "code_snapshot": str(CODE.relative_to(ROOT)), "code_manifest_sha256": sha(CODE / "manifest.json"),
              "protocol_sha256": sha(CODE / "protocol.json"), "data_manifest_sha256": sha(OUT / "data_manifest.json"),
              "split_audit_sha256": sha(OUT / "split_audit.json"), "model_seed": 123, "stochastic_samples": 8,
              "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"}, "jobs": {}}
    ledger_path = OUT / "job_ledger.json"; ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    smoke = submit("smoke.pbs"); ledger["jobs"]["smoke"] = smoke; ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ae = submit("ae.pbs", smoke); ledger["jobs"]["ae"] = ae; ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    dynamics = submit("dynamics.pbs", ae); ledger["jobs"]["dynamics"] = dynamics; ledger["status"] = "submitted"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n"); print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__": main()
