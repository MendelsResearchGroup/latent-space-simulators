#!/usr/bin/env python3
"""Freeze and submit the full 200×1500 mini-data conversion."""
from pathlib import Path
import hashlib, json, shutil, subprocess, re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "hpc/experiments/mini_stiffness"
OUT = ROOT / "notebooks/results/mini_stiffness"
CODE = OUT / "full200_conversion_code_v1"
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""): digest.update(block)
    return digest.hexdigest()
if CODE.exists(): raise FileExistsError(CODE)
CODE.mkdir(parents=True)
for path in (ROOT / "tools/data/convert_data_mini_full.py", ROOT / "tools/data/convert_data_mini.py", HERE / "convert_full200.pbs"):
    shutil.copyfile(path, CODE / path.name)
hashes = {"full_converter": sha(CODE / "convert_data_mini_full.py"), "base_converter": sha(CODE / "convert_data_mini.py"),
          "pbs": sha(CODE / "convert_full200.pbs"), "archive": sha(ROOT / "data/data_mini.tar.gz"),
          "prefix_manifest": sha(ROOT / "data/mini_data_converted.manifest.json")}
(CODE / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
command = ["qsub", "-v", f"LSS_CONVERSION_CODE={CODE}", str(CODE / "convert_full200.pbs")]
job = subprocess.check_output(command, text=True).strip()
if not re.fullmatch(r"\d+\.[\w.-]+", job): raise ValueError(job)
record = {"job_id": job, "submitted_at": datetime.now(timezone.utc).isoformat(), "command": command,
          "code_dir": str(CODE), "hashes": hashes, "scope": "node 200×500; stiffness 200×1500 full raw frames"}
with (OUT / "full200_conversion_jobs.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
print(json.dumps(record), flush=True)
