#!/usr/bin/env python3
"""Create a frozen snapshot and submit smoke-gated size-weighted AE fits only on --submit."""
from __future__ import annotations

import argparse, hashlib, json, os, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/rg/mendels_prj/alexander.z/latent-space-simulators")
HERE = ROOT / "hpc/experiments/mini_temperature200_frames500_size_weighted"
OUT = ROOT / "notebooks/results/mini_temperature200_frames500_size_weighted/v2"
PYTHON = "/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python"
ARMS, SEEDS = ("uniform", "inverse_energy_cap100"), (123, 456, 786)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qsub(command: list[str]) -> str:
    job = subprocess.check_output(command, text=True).strip()
    assert re.fullmatch(r"\d+\.[\w.-]+", job), job
    return job


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = {"arms": ARMS, "seeds": SEEDS, "smoke_gate": "each production fit depends afterok on its own smoke fit",
            "submission": bool(args.submit)}
    if not args.submit:
        print(json.dumps(plan, indent=2)); return
    if OUT.exists(): raise FileExistsError(f"immutable study output already exists: {OUT}")
    env = dict(os.environ, LSS_PROJECT_ROOT=str(ROOT), OMP_NUM_THREADS="1", MPLCONFIGDIR="/tmp/mt200-size-weighted-submit")
    subprocess.run([PYTHON, str(HERE / "train.py"), "--assert-config"], check=True, env=env)
    # Exercise the actual mixture split resolver on the prepared source before snapshot/qsub.
    subprocess.run([PYTHON, str(HERE / "train.py"), "--seed", "123", "--smoke", "--preflight-splits"], check=True, env=env)
    OUT.mkdir(parents=True); code = OUT / "code_v2"; code.mkdir()
    shutil.copytree(ROOT / "src/lss", code / "src/lss", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("train.py", "fit.pbs", "protocol.json", "submit.py"):
        shutil.copy2(HERE / name, code / name)
    shutil.copy2(ROOT / "notebooks/results/mini_temperature200_frames500/data_manifest.json", code / "data_manifest.json")
    files = {str(path.relative_to(code)): sha(path) for path in code.rglob("*") if path.is_file()}
    (code / "manifest.json").write_text(json.dumps({"files": files}, indent=2, sort_keys=True) + "\n")
    ledger = {"study": "mini_temperature200_frames500_size_weighted", "status": "submitting", "snapshot": str(code.relative_to(ROOT)),
              "code_manifest_sha256": sha(code / "manifest.json"), "protocol_sha256": sha(code / "protocol.json"),
              "data_manifest_sha256": sha(code / "data_manifest.json"), "jobs": {}, "submitted_at": datetime.now(timezone.utc).isoformat(),
              "resources": {"queue": "mendels_q", "ncpus": 8, "mem": "16gb", "place": "free:shared"},
              "supersedes": {"snapshot": "notebooks/results/mini_temperature200_frames500_size_weighted/v1/code_v1",
                  "failure": "all six smoke fits raised KeyError: train_count before training",
                  "failed_smoke_jobs": ["4687498.zeus-master", "4687500.zeus-master", "4687502.zeus-master", "4687504.zeus-master", "4687506.zeus-master", "4687508.zeus-master"],
                  "dependent_fit_jobs": ["4687499.zeus-master", "4687501.zeus-master", "4687503.zeus-master", "4687505.zeus-master", "4687507.zeus-master", "4687509.zeus-master"]}}
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"; env_value = f"LSS_CODE_ROOT={code},LSS_ARM={arm},LSS_SEED={seed},LSS_SMOKE_FLAG=--smoke"
            smoke = qsub(["qsub", "-N", "m2sw_smoke", "-v", env_value, str(code / "fit.pbs")])
            production = qsub(["qsub", "-N", "m2sw_prod", "-W", f"depend=afterok:{smoke}", "-v",
                               f"LSS_CODE_ROOT={code},LSS_ARM={arm},LSS_SEED={seed}", str(code / "fit.pbs")])
            ledger["jobs"][key] = {"smoke": smoke, "fit": production}
    ledger["status"] = "submitted"
    (OUT / "job_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__": main()
