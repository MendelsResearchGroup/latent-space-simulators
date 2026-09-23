import hashlib,json,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators');HERE=ROOT/'hpc/experiments/mini_temperature1000_linear_prefix';OUT=ROOT/'notebooks/results/mini_temperature1000_linear_prefix';CODE=OUT/'code_v1';ARMS=('joint_low_uniform_retry1','node_low_uniform_retry1')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def main():
 if OUT.exists():raise FileExistsError(OUT)
 env={'LSS_PROJECT_ROOT':str(ROOT),'PYTHONPATH':f'{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src','OMP_NUM_THREADS':'1'}
 subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/'run.py'),'--assert-config'],check=True,env=env)
 CODE.mkdir(parents=True);shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ('run.py','pipeline.pbs','submit.py'):shutil.copy2(HERE/name,CODE/name)
 shutil.copy2(ROOT/'notebooks/results/mini_temperature1000_linear_fit/code_v2/evaluate.py',CODE/'evaluate.py')
 files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};write(CODE/'manifest.json',{'files':files})
 ledger={'study':'mini_temperature1000_linear_prefix','snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'cohort':'mixed only; direct full200 test IDs40:119; seed123','method':'post-fit prefix affine normalized boundary strain OLS frames30..horizon, horizon37..199; t199 parity guard only','jobs':{},'status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat()};lp=OUT/'job_ledger.json';write(lp,ledger)
 for arm in ARMS:
  job=subprocess.check_output(['qsub','-N','mini_t1000_prefix','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(CODE/'pipeline.pbs')],text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',job);ledger['jobs'][arm]=job;write(lp,ledger)
 ledger['status']='submitted';write(lp,ledger);print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
