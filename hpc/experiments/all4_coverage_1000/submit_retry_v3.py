#!/usr/bin/env python3
"""Freeze and submit the exposure-only v3 recovery after failed v2 jobs."""
import hashlib,json,os,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
HERE=ROOT/'hpc/experiments/all4_coverage_1000'
OUT=ROOT/'notebooks/results/all4_coverage_1000'
CODE=OUT/'code_v3'
ARMS=('all4_one_each','all4_twenty_each')
TAG='_retry_v3'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def qsub(script,arm,dependency=None):
 command=['qsub','-N','all4cov1000v3']
 if dependency:command+=['-W','depend='+dependency]
 command+=['-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm},LSS_TAG={TAG}',str(script)]
 job=subprocess.check_output(command,text=True).strip()
 assert re.fullmatch(r'\d+\.[\w.-]+',job),job
 return job

def main():
 ledger_path=OUT/'job_ledger.json';ledger=json.loads(ledger_path.read_text())
 assert ledger['status']=='submitted' and not CODE.exists()
 assert sha(OUT/'data_manifest.json')==ledger['data_manifest_sha256']
 environment=dict(os.environ,LSS_PROJECT_ROOT=str(ROOT),PYTHONPATH=f'{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src',OMP_NUM_THREADS='1')
 for script,args in [('train.py',['--assert-config']),('linear_run.py',['--assert-config'])]:
  subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/script),*args],check=True,env=environment)
 CODE.mkdir();shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ('train.py','collect.py','evaluate.py','linear_run.py','thermal_response.py','thermal_metric.py','pipeline.pbs','collect.pbs','protocol.json','prepare_data_manifest.py','submit_retry_v3.py'):
  shutil.copy2(HERE/name,CODE/name)
 shutil.copy2(OUT/'data_manifest.json',CODE/'data_manifest.json')
 files={str(path.relative_to(CODE)):sha(path) for path in CODE.rglob('*') if path.is_file()}
 (CODE/'manifest.json').write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
 retry={'tag':TAG,'snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'data_manifest_sha256':sha(CODE/'data_manifest.json'),'reason':'v2 exposure export wrongly zipped a flat interleaved source mixture; no model/data/objective/selection change','fixed_in':'train.py source-wise exposure filtering with exact split/source/ID coverage assertion','resources':ledger['resources'],'submitted_at':datetime.now(timezone.utc).isoformat(),'jobs':{}}
 ledger.setdefault('retries',[]).append(retry);ledger_path.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 for arm in ARMS:
  retry['jobs'][arm]={'fit':qsub(CODE/'pipeline.pbs',arm)};ledger_path.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 for arm in ARMS:
  retry['jobs'][arm]['collector']=qsub(CODE/'collect.pbs',arm,'afterok:'+retry['jobs'][arm]['fit']);ledger_path.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 print(json.dumps(retry,indent=2,sort_keys=True))

if __name__=='__main__':main()
