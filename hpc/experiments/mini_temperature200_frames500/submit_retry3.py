#!/usr/bin/env python3
"""Repair smoke collection, then gate production fits and post-fit analyses."""
from __future__ import annotations

import hashlib, json, os, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
HERE=ROOT/'hpc/experiments/mini_temperature200_frames500'
OUT=ROOT/'notebooks/results/mini_temperature200_frames500'
CODE=OUT/'code_v6'
LEDGER=OUT/'job_ledger.json'
PYTHON='/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def submit(command):
 job=subprocess.check_output(command,text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',job),job;return job

def main():
 ledger=json.loads(LEDGER.read_text());assert not CODE.exists()
 env=dict(os.environ,LSS_PROJECT_ROOT=str(ROOT),OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mt200-r3-submit',PYTHONPATH=f'{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src')
 subprocess.run([PYTHON,str(HERE/'train.py'),'--assert-config'],check=True,env=env)
 CODE.mkdir();shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 names=('train.py','evaluate.py','collect.py','smoke_collect.pbs','fit_production.pbs','collect.pbs','morph.py','morph.pbs','attribution.py','attribution.pbs','protocol.json','submit_retry3.py')
 for name in names:shutil.copy2(HERE/name,CODE/name)
 files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};(CODE/'manifest.json').write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
 protocol=json.loads((CODE/'protocol.json').read_text());jobs={'smoke_collectors':{},'base':{},'morph_reuse':{}}
 for arm in protocol['base_arms']:
  jobs['smoke_collectors'][arm]=submit(['qsub','-N',f'm2s6_{arm[:6]}','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(CODE/'smoke_collect.pbs')])
 for arm in protocol['base_arms']:
  gate=jobs['smoke_collectors'][arm]
  fit=submit(['qsub','-N',f'm2p6_{arm[:6]}','-W',f'depend=afterok:{gate}','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(CODE/'fit_production.pbs')])
  collect=submit(['qsub','-N',f'm2c6_{arm[:6]}','-W',f'depend=afterok:{fit}','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(CODE/'collect.pbs')])
  jobs['base'][arm]={'fit':fit,'collector':collect}
 for count,arm in ((1,'all4_one_each'),(20,'all4_twenty_each')):
  dependency=jobs['base'][arm]['fit']
  jobs['morph_reuse'][f'count{count}_seed123']=submit(['qsub','-W',f'depend=afterok:{dependency}','-v',f'LSS_CODE_ROOT={CODE},LSS_COUNT={count},LSS_SEED=123,LSS_REUSE_FLAG=--reuse',str(CODE/'morph.pbs')])
 dependencies=':'.join(jobs['base'][a]['fit'] for a in ('all4_one_each','all4_twenty_each','stiffness_low_only','joint_low'))
 jobs['attribution']=submit(['qsub','-W',f'depend=afterok:{dependencies}','-v',f'LSS_CODE_ROOT={CODE}',str(CODE/'attribution.pbs')])
 record={'status':'submitted','reason':'post-smoke evaluator-only repair; reuse completed smoke checkpoints; production-only fits gated on corrected smoke collectors','submitted_at':datetime.now(timezone.utc).isoformat(),'snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'data_manifest_sha256':sha(OUT/'data_manifest.json'),'protocol_sha256':sha(CODE/'protocol.json'),'jobs':jobs,'resources_each':{'queue':'mendels_q','placement':'place=free:shared','ncpus':8,'memory':'16gb'}}
 ledger['status']='retry3_smoke_repair_submitted';ledger.setdefault('retries',[]).append(record);LEDGER.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n');print(json.dumps(record,indent=2,sort_keys=True))

if __name__=='__main__':main()
