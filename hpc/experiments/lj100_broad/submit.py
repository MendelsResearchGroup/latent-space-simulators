#!/usr/bin/env python3
"""Freeze the broad study, run its smoke gate, then submit the declared matrix."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT/'hpc/experiments/lj100_broad'
OUT = ROOT/'notebooks/results/lj100_broad'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def freeze(version):
    code = OUT/version
    if not code.exists():
        code.mkdir(parents=True)
        for name in ('run.py','evaluate.py','prepare.py','submit.py','prepare.pbs','run.pbs','collect.pbs'):
            shutil.copy2(HERE/name,code/name)
        shutil.copy2(OUT/'protocol.json',code/'protocol.json')
        shutil.copy2(ROOT/'tools/analysis/collect_lj100_broad.py',code/'collect.py')
        shutil.copytree(ROOT/'src/lss',code/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        manifest = {str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()}
        (code/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    manifest = json.loads((code/'manifest.json').read_text())
    assert {k:sha(code/k) for k in manifest}==manifest
    assert sha(OUT/'protocol.json')==sha(code/'protocol.json')
    return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('smoke','production'),required=True)
    parser.add_argument('--version',required=True)
    args = parser.parse_args()
    code = freeze(args.version)
    ledger = OUT/'jobs.jsonl'
    old = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    protocol = json.loads((code/'protocol.json').read_text())

    def submit(identity, pbs, variables, dependencies):
        variables = {key: str(value) for key, value in variables.items()}
        matching = [r for r in old if r['identity']==identity and r['snapshot']==str(code)]
        if matching:
            return matching[0]['job_id']
        cmd = ['qsub','-v',','.join(f'{k}={v}' for k,v in variables.items())]
        if dependencies:
            cmd += ['-W',dependencies]
        if args.stage == 'production' and pbs == 'run.pbs':
            cmd += ['-l', 'select=1:ncpus=4:mem=4gb']
        cmd += [str(code/pbs)]
        output = subprocess.check_output(cmd,text=True).strip()
        job = output.splitlines()[-1]
        assert re.fullmatch(r'\d+\.[\w.-]+',job), output
        row = dict(identity=identity,job_id=job,snapshot=str(code),variables=variables,dependencies=dependencies,qsub_command=cmd,
                   submitted_at=datetime.now(timezone.utc).isoformat(),submitter_sha256=sha(Path(__file__)))
        with ledger.open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
        old.append(row)
        print(identity,job,flush=True)
        return job

    if args.stage=='production':
        evidence = []
        for family in protocol['families']:
            for context in protocol['context_modes']:
                path = OUT/'runs'/f'{family}_{context}_s123_smoke_{code.name}'
                done = json.loads((path/'completed.json').read_text())
                recipe = json.loads((path/'recipe.json').read_text())
                assert done['status']=='completed' and done['rows']==600
                assert recipe['frozen_before']==recipe['frozen_after']
                assert recipe['checks']['pure_position_gradient']>0
                assert recipe['code_manifest_sha256']==sha(code/'manifest.json')
                evidence.append(dict(family=family,context=context,job_id=done['job_id'],checks=recipe['checks']))
        (OUT/'smoke_gate.json').write_text(json.dumps(dict(code_manifest_sha256=sha(code/'manifest.json'),evidence=evidence),indent=2)+'\n')
    seeds = [123] if args.stage=='smoke' else [123,456,786]
    cache_paths, cache_jobs = {}, {}
    for seed in seeds:
        filename = f'smoke_s{seed}_{code.name}.pt' if args.stage=='smoke' else f's{seed}.pt'
        cache_paths[seed] = OUT/'cache'/filename
        cache_jobs[seed] = submit(f'{args.stage}_cache_s{seed}','prepare.pbs',
            dict(LSS_CODE_ROOT=code,LSS_SEED=seed,LSS_CACHE=cache_paths[seed],LSS_SMOKE=int(args.stage=='smoke')),
            '')
    jobs = []
    for arm in protocol['matrix']:
        if arm['seed'] not in seeds:
            continue
        seed = arm['seed']
        jobs.append(submit(f'{args.stage}_{arm["run_name"]}','run.pbs',
            dict(LSS_CODE_ROOT=code,LSS_SEED=seed,LSS_CACHE=cache_paths[seed],LSS_SMOKE=int(args.stage=='smoke'),
                 LSS_FAMILY=arm['family'],LSS_CONTEXT=arm['context_mode']),
            'depend=afterok:'+cache_jobs[seed]))
    if args.stage=='production':
        submit('production_collector','collect.pbs',dict(LSS_CODE_ROOT=code),'depend=afterany:'+':'.join(jobs))


if __name__=='__main__':
    main()
