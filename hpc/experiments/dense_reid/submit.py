#!/usr/bin/env python3
"""Freeze one harness and submit smoke or the six predeclared dense Reid cells."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'hpc/experiments/dense_reid'
OUT=ROOT/'notebooks/results/dense_reid'
CODE=OUT/'code'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def freeze():
    if CODE.exists():
        hashes=json.loads((CODE/'hashes.json').read_text())
        assert all(sha(CODE/k)==v for k,v in hashes.items()), 'Frozen code changed'
        return
    CODE.mkdir()
    shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('run.py','evaluate.py','run.pbs','submit.py'):
        shutil.copyfile(HERE/name,CODE/name)
    hashes={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()}
    (CODE/'hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('smoke','full'));a=p.parse_args()
    assert (OUT/'prepared.json').is_file()
    freeze()
    ledger=OUT/'jobs.jsonl'
    old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    if a.stage=='full':
        assert json.loads((OUT/'runs/d4_s123_smoke/completed.json').read_text())['status']=='completed'
    cells=[(4,123)] if a.stage=='smoke' else [(d,s) for d in (2,4) for s in (123,456,786)]
    for d,s in cells:
        if any(r['stage']==a.stage and r['dimension']==d and r['seed']==s for r in old):
            continue
        env=f'LSS_CODE_ROOT={CODE},LSS_DIMENSION={d},LSS_SEED={s},LSS_SMOKE={int(a.stage=="smoke")}'
        cmd=['qsub','-v',env,'-N',f'dense_d{d}_{s}',str(CODE/'run.pbs')]
        job=subprocess.check_output(cmd,text=True).strip()
        assert re.fullmatch(r'\d+\.\S+',job),repr(job)
        row={'stage':a.stage,'dimension':d,'seed':s,'job_id':job,'command':cmd,
             'submitted_at':datetime.now(timezone.utc).isoformat(),
             'code_manifest_sha256':sha(CODE/'hashes.json'),'data_manifest_sha256':sha(OUT/'prepared.json')}
        with ledger.open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
