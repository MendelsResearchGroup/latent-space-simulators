#!/usr/bin/env python3
"""Freeze the mini-data converter and submit its single PBS conversion job."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "hpc/experiments/mini_stiffness"
OUT = ROOT / "notebooks/results/mini_stiffness"
CODE = OUT / "conversion_code_v2"

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()

if CODE.exists():
    raise FileExistsError(f"frozen converter already exists: {CODE}")
CODE.mkdir(parents=True)
converter = ROOT / "tools/data/convert_data_mini.py"
pbs = HERE / "convert.pbs"
shutil.copyfile(converter, CODE / converter.name)
shutil.copyfile(pbs, CODE / pbs.name)
hashes = {"converter": sha(CODE / converter.name), "pbs": sha(CODE / pbs.name),
          "legacy_network": sha(Path("/rg/mendels_prj/alexander.z/MetaForge/src/network.py")),
          "archive": sha(ROOT / "data/data_mini.tar.gz")}
(CODE / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
environment = ",".join((f"LSS_CONVERTER={CODE / converter.name}", f"LSS_CONVERTER_SHA256={hashes['converter']}",
                        f"LSS_CONVERTER_PBS_SHA256={hashes['pbs']}", f"LSS_LEGACY_NETWORK_SHA256={hashes['legacy_network']}"))
command = ["qsub", "-v", environment, str(CODE / pbs.name)]
job = subprocess.check_output(command, text=True).strip()
if not re.fullmatch(r"\d+\.[\w.-]+", job):
    raise ValueError(job)
record = {"version": 2, "job_id": job, "submitted_at": datetime.now(timezone.utc).isoformat(), "command": command,
          "code_dir": str(CODE), "hashes": hashes}
with (OUT / "conversion_jobs.jsonl").open("a") as stream:
    stream.write(json.dumps(record) + "\n")
print(json.dumps(record), flush=True)
