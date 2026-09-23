#!/usr/bin/env python3
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
HERE = ROOT / "hpc/experiments/mini_temperature1000_initial_anchor_pilot"
OUT = ROOT / "notebooks/results/mini_temperature1000_initial_anchor_pilot"
CODE = OUT / "code_v1"
BASE_CODE = ROOT / "notebooks/results/mini_temperature1000_mixed_only/code_v2/src/lss"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(script, dependency=None):
    command = ["qsub"]
    if dependency:
        command += ["-W", "depend=afterok:" + dependency]
    command += ["-v", f"LSS_CODE_ROOT={CODE}", str(script)]
    job = subprocess.check_output(command, text=True).strip()
    if not re.fullmatch(r"\d+\.[\w.-]+", job):
        raise RuntimeError(job)
    return job


def main():
    env = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), OMP_NUM_THREADS="1")
    subprocess.run([
        "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python",
        str(HERE / "train.py"), "--assert-config"
    ], check=True, env={**env, "PYTHONPATH": f"{ROOT}/notebooks/results/mini_temperature1000_mixed_only/code_v2/src:{HERE}"})
    if CODE.exists():
        raise FileExistsError(CODE)
    CODE.mkdir(parents=True)
    shutil.copytree(BASE_CODE, CODE / "src/lss")
    for name in ("anchor_context.py", "train.py", "evaluate.py", "run.pbs", "collect.pbs", "submit.py"):
        shutil.copy2(HERE / name, CODE / name)
    files = {str(path.relative_to(CODE)): sha(path) for path in CODE.rglob("*") if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    ledger = {
        "study": "mini_temperature1000_initial_anchor_pilot",
        "status": "submitting",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "explicit frame-zero latent anchoring improves mixed-T stiffness rollout",
        "baseline": "mini_temperature1000_mixed_only/joint_mixed_only seed123",
        "intervention": "append frame-zero latent to existing static context at every delta step",
        "unchanged": ["frozen AE", "split", "seed", "20 trajectories/source", "frames0:199", "delta MLP", "one-step loss", "optimizer", "validation latent-loss selection"],
        "evaluation": "heldout development IDs40:69; post-fit OLS frames30:199 primary; endpoint199 secondary",
        "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
        "snapshot": str(CODE.relative_to(ROOT)),
        "code_manifest_sha256": sha(CODE / "manifest.json"),
        "jobs": {},
    }
    OUT.mkdir(exist_ok=True)
    ledger_path = OUT / "job_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    fit = qsub(CODE / "run.pbs")
    ledger["jobs"]["fit"] = fit
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    collector = qsub(CODE / "collect.pbs", fit)
    ledger["jobs"]["collector"] = collector
    ledger["jobs"]["collector_dependency"] = "afterok:" + fit
    ledger["status"] = "submitted"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()

