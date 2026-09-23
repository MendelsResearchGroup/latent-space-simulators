#!/usr/bin/env python3
"""Queue three distinct shared dynamics hypotheses and post-fit mixed-T transfer."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'hpc/experiments/lj100_radical'
OUT=ROOT/'notebooks/results/lj100_radical'


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
        shutil.copy2(ROOT/'tools/analysis/collect_lj100_radical.py',code/'collect.py')
        shutil.copytree(ROOT/'src/lss',code/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (code/'manifest.json').write_text(json.dumps({str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()},indent=2)+'\n')
    manifest=json.loads((code/'manifest.json').read_text())
    assert {k:sha(code/k) for k in manifest}==manifest
    assert sha(code/'protocol.json')==sha(OUT/'protocol.json')
    protocol=json.loads((code/'protocol.json').read_text())
    ledger=OUT/'jobs.jsonl'
    records=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []

    def submit(identity,pbs,variables,dependency):
        existing=[row for row in records if row['identity']==identity and row['snapshot']==str(code)]
        if existing:
            return existing[0]['job_id']
        variables={k:str(v) for k,v in variables.items()}
        cmd=['qsub','-v',','.join(f'{k}={v}' for k,v in variables.items())]
        if dependency:
            cmd += ['-W',dependency]
        cmd.append(str(code/pbs))
        output=subprocess.check_output(cmd,text=True).strip()
        job=output.splitlines()[-1]
        assert re.fullmatch(r'\d+\.[\w.-]+',job),output
        row=dict(identity=identity,job_id=job,snapshot=str(code),variables=variables,dependency=dependency,
                 qsub_command=cmd,submitter_sha256=sha(Path(__file__)),submitted_at=datetime.now(timezone.utc).isoformat())
        with ledger.open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
        records.append(row)
        print(identity,job,flush=True)
        return job

    if args.stage=='production':
        evidence=[]
        for family in protocol['families']:
            folder=OUT/'runs'/f'{family}_attention_s123_smoke_{code.name}'
            done=json.loads((folder/'completed.json').read_text())
            recipe=json.loads((folder/'recipe.json').read_text())
            transfer=json.loads((folder/'mixed_completed.json').read_text())
            assert done['status']=='completed' and done['rows']==600
            assert transfer['status']=='completed' and transfer['rows']==4000
            assert recipe['checks']['pure_position_gradient']>0
            assert recipe['frozen_before']==recipe['frozen_after']
            assert recipe['code_manifest_sha256']==sha(code/'manifest.json')
            evidence.append(dict(family=family,job_id=done['job_id'],checks=recipe['checks']))
        (OUT/'smoke_gate.json').write_text(json.dumps(evidence,indent=2)+'\n')
    seeds=[123] if args.stage=='smoke' else [123,456,786]
    fits=[]
    for arm in protocol['matrix']:
        if arm['seed'] not in seeds:
            continue
        seed=arm['seed']
        filename='smoke_s123_code_v1.pt' if args.stage=='smoke' else f's{seed}.pt'
        cache=ROOT/'notebooks/results/lj100_broad/cache'/filename
        assert cache.exists() and cache.with_suffix('.completed.json').exists()
        fits.append(submit(args.stage+'_'+arm['run_name'],'run.pbs',
            dict(LSS_CODE_ROOT=code,LSS_SEED=seed,LSS_FAMILY=arm['family'],LSS_CONTEXT='attention',
                 LSS_CACHE=cache,LSS_SMOKE=int(args.stage=='smoke')),''))
    transfers=[]
    for seed in seeds:
        filename=f'mixed_smoke_s{seed}_{code.name}.pt' if args.stage=='smoke' else f'mixed_s{seed}.pt'
        cache=OUT/'cache'/filename
        prep=submit(args.stage+f'_mixed_cache_s{seed}','mixed_prepare.pbs',
            dict(LSS_CODE_ROOT=code,LSS_SEED=seed,LSS_CACHE=cache,LSS_SMOKE=0),
            'depend=afterany:'+':'.join(fits))
        for family in protocol['families']:
            transfers.append(submit(args.stage+f'_mixed_{family}_s{seed}','mixed_evaluate.pbs',
                dict(LSS_CODE_ROOT=code,LSS_SEED=seed,LSS_FAMILY=family,LSS_SMOKE=int(args.stage=='smoke')),
                'depend=afterok:'+prep))
    if args.stage=='production':
        submit('production_collector','collect.pbs',dict(LSS_CODE_ROOT=code),'depend=afterany:'+':'.join(transfers))


if __name__=='__main__':main()
