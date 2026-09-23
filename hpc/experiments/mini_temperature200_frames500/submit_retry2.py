#!/usr/bin/env python3
"""Submit the complete base and morphology matrix after successful data audit."""
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
CODE = OUT / "code_v5"
LEDGER = OUT / "job_ledger.json"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"


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
    data_manifest = json.loads((OUT / "data_manifest.json").read_text())
    assert data_manifest["status"] == "complete" and data_manifest["cached_prefix_frames"] == 500
    assert data_manifest["pbs_job_id"] == "4687289.zeus-master"
    assert not CODE.exists()
    environment = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mt200-r2-submit", PYTHONPATH=f"{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src")
    subprocess.run([PYTHON, str(HERE / "train.py"), "--assert-config"], check=True, env=environment)
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    names = ("train.py", "evaluate.py", "collect.py", "fit.pbs", "collect.pbs", "morph.py", "morph.pbs", "morph_array.pbs", "morph_matrix.json", "attribution.py", "attribution.pbs", "protocol.json", "prepare.py", "prepare.pbs", "submit_retry2.py")
    for name in names:
        shutil.copy2(HERE / name, CODE / name)
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    protocol = json.loads((CODE / "protocol.json").read_text())
    jobs = {"base": {}, "morph": {}}
    for arm in protocol["base_arms"]:
        fit = accepted(["qsub", "-N", f"m2f5r2_{arm[:6]}", "-v", f"LSS_CODE_ROOT={CODE},LSS_ARM={arm}", str(CODE / "fit.pbs")])
        jobs["base"][arm] = {"fit": fit}
    for arm in protocol["base_arms"]:
        fit = jobs["base"][arm]["fit"]
        jobs["base"][arm]["collector"] = accepted(["qsub", "-N", f"m2e5r2_{arm[:6]}", "-W", f"depend=afterok:{fit}", "-v", f"LSS_CODE_ROOT={CODE},LSS_ARM={arm}", str(CODE / "collect.pbs")])
    jobs["morph"]["fresh_22_cell_array"] = accepted(["qsub", "-J", "0-21", "-v", f"LSS_CODE_ROOT={CODE}", str(CODE / "morph_array.pbs")])
    for count, arm in ((1, "all4_one_each"), (20, "all4_twenty_each")):
        dependency = jobs["base"][arm]["fit"]
        jobs["morph"][f"reuse_count{count}_seed123"] = accepted(["qsub", "-W", f"depend=afterok:{dependency}", "-v", f"LSS_CODE_ROOT={CODE},LSS_COUNT={count},LSS_SEED=123,LSS_REUSE_FLAG=--reuse", str(CODE / "morph.pbs")])
    attribution_dependencies = ":".join(jobs["base"][arm]["fit"] for arm in ("all4_one_each", "all4_twenty_each", "stiffness_low_only", "joint_low"))
    jobs["attribution"] = accepted(["qsub", "-W", f"depend=afterok:{attribution_dependencies}", "-v", f"LSS_CODE_ROOT={CODE}", str(CODE / "attribution.pbs")])
    record = {
        "status": "submitted", "submitted_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Packaging-only retry after robust repository-root discovery; no scientific setting changed.",
        "data_job": data_manifest["pbs_job_id"], "data_manifest_sha256": sha(OUT / "data_manifest.json"),
        "snapshot": str(CODE.relative_to(ROOT)), "code_manifest_sha256": sha(CODE / "manifest.json"),
        "protocol_sha256": sha(CODE / "protocol.json"), "matrix_sha256": sha(CODE / "morph_matrix.json"),
        "jobs": jobs, "base_arms": len(protocol["base_arms"]), "morph_cells": 24, "fresh_morph_fits": 22,
        "attribution": "four-model response-blind feature and physical-edge gradients on fixed stiffness-low ID40, frames0-5/20/50",
        "resources_each": {"queue": "mendels_q", "placement": "place=free:shared", "ncpus": 8, "memory": "16gb", "walltime": "24:00:00"},
    }
    ledger["status"] = "retry2_full_matrix_submitted"
    ledger["data_preparation_retry2"] = {"job": data_manifest["pbs_job_id"], "status": "complete", "data_manifest_sha256": record["data_manifest_sha256"], "prepare_script_sha256": data_manifest["script_sha256"]}
    ledger.setdefault("retries", []).append(record)
    LEDGER.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
