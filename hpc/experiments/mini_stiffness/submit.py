"""Freeze the mini-data experiment and queue smoke then production after conversion."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import re
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'hpc/experiments/mini_stiffness'
OUT=ROOT/'notebooks/results/mini_stiffness'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--conversion-job',required=True);a=p.parse_args()
assert re.fullmatch(r'\d+\.[\w.-]+',a.conversion_job)
code=OUT/'code_v2';code.mkdir()
shutil.copytree(ROOT/'src/lss',code/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('run.py','evaluate.py','pipeline.pbs','check_smoke.py','submit.py'):
    shutil.copyfile(HERE/name,code/name)
shutil.copyfile(OUT/'protocol.json',code/'protocol.json')
(code/'hashes.json').write_text(json.dumps({str(f.relative_to(code)):sha(f) for f in code.rglob('*') if f.is_file()},indent=2)+'\n')
cmd=['qsub','-W','depend=afterok:'+a.conversion_job,'-v','LSS_CODE_ROOT='+str(code),str(code/'pipeline.pbs')]
job=subprocess.check_output(cmd,text=True).strip()
assert re.fullmatch(r'\d+\.[\w.-]+',job),job
record=dict(job_id=job,conversion_job=a.conversion_job,command=cmd,submitted_at=datetime.now(timezone.utc).isoformat(),code_manifest_sha256=sha(code/'hashes.json'),protocol_sha256=sha(code/'protocol.json'))
with (OUT/'jobs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record),flush=True)
