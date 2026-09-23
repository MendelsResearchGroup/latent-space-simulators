#!/usr/bin/env python3
"""Freeze and submit mini-100 data preparation."""
from pathlib import Path
import hashlib, json, shutil, subprocess, re
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[3]
HERE, OUT = ROOT / "hpc/experiments/mini_100", ROOT / "notebooks/results/mini_100"
CODE = OUT / "preparation_code_v1"
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): digest.update(block)
    return digest.hexdigest()
if CODE.exists(): raise FileExistsError(CODE)
CODE.mkdir(parents=True)
for name in ("prepare.py", "prepare.pbs", "submit_prepare.py"): shutil.copyfile(HERE / name, CODE / name)
hashes = {name: sha(CODE / name) for name in ("prepare.py", "prepare.pbs", "submit_prepare.py")}
(CODE / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
environment = f"LSS_PREP_CODE={CODE},LSS_PREP_CODE_SHA256={hashes['prepare.py']},LSS_PREP_PBS_SHA256={hashes['prepare.pbs']}"
command = ["qsub", "-v", environment, str(CODE / "prepare.pbs")]
job = subprocess.check_output(command, text=True).strip()
if not re.fullmatch(r"\d+\.[\w.-]+", job): raise ValueError(job)
record = {"job_id": job, "submitted_at": datetime.now(timezone.utc).isoformat(), "command": command,
          "code_dir": str(CODE), "hashes": hashes, "scope": "120 trajectories/source, exact frames 0..99"}
with (OUT / "prepare_jobs.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
print(json.dumps(record), flush=True)
