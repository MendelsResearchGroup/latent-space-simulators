#!/usr/bin/env python3
"""Freeze and submit the response-blind three-arm pilot."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
OUT = ROOT / "notebooks/results/lj_latent_influence_edges"
CODE = OUT / "code_v2"
FILES = ("protocol.json", "run.py", "run.pbs", "submit.py")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if CODE.exists():
        raise FileExistsError(CODE)
    (CODE / "src/lss/dynamics").mkdir(parents=True)
    for name in FILES:
        shutil.copy2(SOURCE / name, CODE / name)
    shutil.copy2(ROOT / "src/lss/dynamics/latent_influence_gnn.py", CODE / "src/lss/dynamics/latent_influence_gnn.py")
    manifest = {str(path.relative_to(CODE)): sha(path) for path in sorted(CODE.rglob("*")) if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    jobs = []
    smoke = os.environ.get("LSS_SMOKE", "0")
    for seed in (456, 786):
        for arm in ("base", "random", "latent"):
            environment = f"LSS_CODE_ROOT={CODE},LSS_ARM={arm},LSS_SEED={seed},LSS_SMOKE={smoke}"
            result = subprocess.run(["qsub", "-v", environment, str(CODE / "run.pbs")], check=True,
                                    capture_output=True, text=True)
            jobs.append({"arm": arm, "seed": seed, "smoke": smoke == "1", "job_id": result.stdout.strip()})
    (OUT / "jobs_v2.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
