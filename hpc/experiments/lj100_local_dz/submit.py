#!/usr/bin/env python3
"""Submit matched one-frame/three-frame prefix experiments from frozen caches."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'hpc/experiments/lj100_local_dz'
OUT=ROOT/'notebooks/results/lj100_local_dz'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):
            value.update(part)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','production'),required=True)
    parser.add_argument('--version',required=True)
    args=parser.parse_args()
    code=OUT/args.version
    if not code.exists():
        code.mkdir(parents=True)
        for path in HERE.iterdir():
            if path.suffix in ('.py','.pbs'):
                shutil.copy2(path,code/path.name)
        shutil.copy2(OUT/'protocol.json',code/'protocol.json')
        shutil.copy2(ROOT/'tools/analysis/collect_lj100_local_dz.py',code/'collect.py')
        shutil.copytree(ROOT/'src/lss',code/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (code/'manifest.json').write_text(json.dumps({str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()},indent=2)+'\n')
    manifest=json.loads((code/'manifest.json').read_text())
    assert {k:sha(code/k) for k in manifest}==manifest
    assert sha(code/'protocol.json')==sha(OUT/'protocol.json')
    protocol=json.loads((code/'protocol.json').read_text())
    ledger=OUT/'jobs.jsonl'
    records=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []

    def submit(identity,pbs,variables,dependency):
        matching=[r for r in records if r['identity']==identity and r['snapshot']==str(code)]
        if matching:
            return matching[0]['job_id']
        variables={key:str(value) for key,value in variables.items()}
        cmd=['qsub','-v',','.join(f'{key}={value}' for key,value in variables.items())]
        if dependency:
            cmd += ['-W',dependency]
        cmd.append(str(code/pbs))
        output=subprocess.check_output(cmd,text=True).strip()
        job=output.splitlines()[-1]
        assert re.fullmatch(r'\d+\.[\w.-]+',job),output
        record=dict(identity=identity,job_id=job,snapshot=str(code),variables=variables,dependency=dependency,
                    qsub_command=cmd,submitter_sha256=sha(Path(__file__)),submitted_at=datetime.now(timezone.utc).isoformat())
        with ledger.open('a') as stream:
            stream.write(json.dumps(record,sort_keys=True)+'\n')
        records.append(record)
        print(identity,job,flush=True)
        return job

    if args.stage=='production':
        evidence=[]
        for family in protocol['families']:
            for count in protocol['observed_frames']:
                folder=OUT/'runs'/f'{family}_h{count}_s123_smoke_{code.name}'
                done=json.loads((folder/'completed.json').read_text())
                recipe=json.loads((folder/'recipe.json').read_text())
                assert done['status']=='completed' and done['rows']==3*98*5
                assert recipe['checks']['pure_latent_gradient']>0
                assert recipe['frozen_before']==recipe['frozen_after']
                assert recipe['code_manifest_sha256']==sha(code/'manifest.json')
                evidence.append(dict(family=family,observed_frames=count,job_id=done['job_id'],checks=recipe['checks']))
        (OUT/'smoke_gate.json').write_text(json.dumps(evidence,indent=2)+'\n')
    jobs=[]
    for arm in protocol['matrix']:
        if args.stage=='smoke' and arm['seed']!=123:
            continue
        filename='smoke_s123_code_v1.pt' if args.stage=='smoke' else f's{arm["seed"]}.pt'
        cache=ROOT/'notebooks/results/lj100_broad/cache'/filename
        assert cache.exists() and cache.with_suffix('.completed.json').exists()
        jobs.append(submit(args.stage+'_'+arm['run_name'],'run.pbs',
            dict(LSS_CODE_ROOT=code,LSS_SEED=arm['seed'],LSS_FAMILY=arm['family'],LSS_OBSERVED=arm['observed_frames'],
                 LSS_CACHE=cache,LSS_SMOKE=int(args.stage=='smoke')),''))
    if args.stage=='production':
        submit('production_collector','collect.pbs',dict(LSS_CODE_ROOT=code),'depend=afterany:'+':'.join(jobs))


if __name__=='__main__':main()
