import hashlib,json,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators');HERE=ROOT/'hpc/experiments/mini_temperature1000_linear_fit';OUT=ROOT/'notebooks/results/mini_temperature1000_linear_fit';CODE=OUT/'code_v1';ARMS=('joint_low_uniform_retry1','node_low_uniform_retry1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 env={'LSS_PROJECT_ROOT':str(ROOT),'PYTHONPATH':f'{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src','OMP_NUM_THREADS':'1'}
 subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/'run.py'),'--assert-config'],check=True,env=env)
 CODE.mkdir(parents=True);shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n in ('run.py','evaluate.py','thermal_response.py','thermal_metric.py','pipeline.pbs','submit.py'):shutil.copy2(HERE/n,CODE/n)
 files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};(CODE/'manifest.json').write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
 ledger={'study':'mini_temperature1000_linear_fit','snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'cohort':'mixed only; direct full200 IDs40:119; joint=node_mixed+stiffness_mixed; node-only=node_mixed','endpoint_parity':'historical physical fixed-frame0 q0.10 at frame199 recomputed only as parity guard','method':'post-fit normalized boundary strain OLS frames30..199 (all170 finite/nondegenerate x); time-OLS auxiliary comparison only','jobs':{},'status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat()};lp=OUT/'job_ledger.json';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 for arm in ARMS:
  j=subprocess.check_output(['qsub','-N','mini_t1000_linear','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(CODE/'pipeline.pbs')],text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',j);ledger['jobs'][arm]=j;lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 ledger['status']='submitted';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n');print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
