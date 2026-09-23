#!/usr/bin/env python3
"""Freeze and submit the first-50-frame matched training ablation."""
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
SOURCE = ROOT / "hpc/experiments/mini_temperature200_frames500"
HERE = ROOT / "hpc/experiments/mini_temperature200_train50"
OUT = ROOT / "notebooks/results/mini_temperature200_train50"
CODE = OUT / "code_v1"
ORIGINAL_DATA = ROOT / "notebooks/results/mini_temperature200_frames500/data_manifest.json"
RESULT_DIR = "notebooks/results/mini_temperature200_train50"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"
ARMS = ("all4_one_each", "all4_twenty_each")
SEEDS = (123, 456, 786)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submit(command: list[str]) -> str:
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job), job
    return job


def main() -> None:
    assert not OUT.exists(), OUT
    OUT.mkdir(parents=True)
    CODE.mkdir()
    shutil.copytree(ROOT / "src/lss", CODE / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "evaluate.py", "collect.py", "fit_production.pbs", "collect.pbs"):
        shutil.copy2(SOURCE / name, CODE / name)
    shutil.copy2(HERE / "smoke.pbs", CODE / "smoke.pbs")
    shutil.copy2(HERE / "protocol.json", CODE / "protocol.json")
    shutil.copy2(HERE / "submit.py", CODE / "submit.py")
    shutil.copy2(ORIGINAL_DATA, OUT / "data_manifest.json")
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")

    environment = dict(
        os.environ,
        LSS_PROJECT_ROOT=str(ROOT), LSS_RESULT_DIR=RESULT_DIR, LSS_TRAIN_FRAMES="50",
        LSS_EXPORT_ALL_LATENTS="1", OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mt200-t50-submit",
        PYTHONPATH=f"{CODE}:{CODE/'src'}:/rg/mendels_prj/alexander.z/MetaForge/src",
    )
    subprocess.run([PYTHON, str(CODE / "train.py"), "--assert-config"], check=True, env=environment)
    common = (
        f"LSS_CODE_ROOT={CODE},LSS_RESULT_DIR={RESULT_DIR},LSS_TRAIN_FRAMES=50,"
        "LSS_EXPORT_ALL_LATENTS=1"
    )
    smoke = submit(["qsub", "-v", common, str(CODE / "smoke.pbs")])
    jobs = {"smoke": smoke, "runs": {}}
    for arm in ARMS:
        for seed in SEEDS:
            tag = f"_s{seed}"
            values = f"{common},LSS_ARM={arm},LSS_SEED={seed},LSS_TAG={tag}"
            fit = submit(["qsub", "-N", f"t50_{arm[:5]}_{seed}", "-W", f"depend=afterok:{smoke}",
                          "-v", values, str(CODE / "fit_production.pbs")])
            collect = submit(["qsub", "-N", f"t50c_{arm[:5]}_{seed}", "-W", f"depend=afterok:{fit}",
                              "-v", values, str(CODE / "collect.pbs")])
            jobs["runs"][f"{arm}_s{seed}"] = {"fit": fit, "collector": collect}
    record = {
        "status": "submitted", "submitted_at": datetime.now(timezone.utc).isoformat(),
        "study": "mini_temperature200_train50", "training_frames": [0, 49],
        "evaluation_frames": [0, 499], "code_manifest_sha256": sha(CODE / "manifest.json"),
        "protocol_sha256": sha(CODE / "protocol.json"),
        "source_data_manifest_sha256": sha(ORIGINAL_DATA), "jobs": jobs,
        "resources_each": {"queue": "mendels_q", "placement": "place=free:shared", "ncpus": 8, "memory": "16gb"},
    }
    (OUT / "job_ledger.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
