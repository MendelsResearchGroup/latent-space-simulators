#!/usr/bin/env python3
"""Queue final verification after the global freeze; never run physics early."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import hashlib

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'notebooks/results/planar_engineering/study_v1'
CODE=OUT/'verification_code'
HERE=Path(__file__).resolve().parent
CODE.mkdir(exist_ok=False)
for name in ['verify.py','render.py','collect.py','run_final.pbs']:
    shutil.copyfile(HERE/name,CODE/name)
old=ROOT/'notebooks/results/thermal_adam30/study_v1/code'
shutil.copyfile(old/'verify.py',CODE/'thermal_dynamics.py')
shutil.copyfile(old/'thermal_metric.py',CODE/'thermal_metric.py')
(CODE/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.iterdir()},indent=2)+'\n')
recipe=json.loads((OUT/'cohort_and_recipe.json').read_text())
first=json.loads((OUT/'submission_status.json').read_text())
jobs={};record={'freeze_dependency':first['freeze'],'final':jobs}
path=OUT/'final_submission_status.json'
for row in recipe['rows']:
    env=f"LSS_STAGE=final,LSS_OUTPUT={OUT},LSS_COHORT={row['cohort']},LSS_INDEX={row['index']}"
    command=['qsub','-q','mendels_q','-l','select=1:ncpus=1:mem=4gb','-l','place=free:shared','-l','walltime=02:00:00',
             '-W',f"depend=afterok:{first['freeze']}",'-v',env,'-N',f"pfin_{row['index']}",str(CODE/'run_final.pbs')]
    job=subprocess.check_output(command,text=True).strip();assert re.fullmatch(r'\d+\.\S+',job)
    jobs[f"{row['cohort']}_{row['index']}"]=job
    path.write_text(json.dumps(record,indent=2)+'\n')
command=['qsub','-q','mendels_q','-l','select=1:ncpus=1:mem=4gb','-l','place=free:shared','-l','walltime=00:30:00',
         '-W','depend=afterany:'+':'.join(jobs.values()),'-v',f'LSS_STAGE=collect,LSS_OUTPUT={OUT}','-N','planar_collect',str(CODE/'run_final.pbs')]
job=subprocess.check_output(command,text=True).strip();assert re.fullmatch(r'\d+\.\S+',job)
record['collect']=job;path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
