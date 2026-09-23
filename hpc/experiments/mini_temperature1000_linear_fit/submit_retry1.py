"""Schema-only retry for v1 endpoint-parity column-access failure."""
import hashlib,json,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
HERE=ROOT/'hpc/experiments/mini_temperature1000_linear_fit'
OUT=ROOT/'notebooks/results/mini_temperature1000_linear_fit'
CODE=OUT/'code_v2';ARMS=('joint_low_uniform_retry1','node_low_uniform_retry1');TAG='retry1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 if CODE.exists():raise FileExistsError(CODE)
 for arm in ARMS:
  for suffix in (f'_{TAG}_smoke',f'_{TAG}'):
   if (OUT/(arm+suffix)).exists():raise FileExistsError(OUT/(arm+suffix))
 env={'LSS_PROJECT_ROOT':str(ROOT),'PYTHONPATH':f'{HERE}:{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src','OMP_NUM_THREADS':'1'}
 subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/'run.py'),'--assert-config'],check=True,env=env)
 old=json.load(open(OUT/'job_ledger.json'));old['status']='failed';old['failure']={'jobs':old['jobs'],'cause':'schema-only endpoint parity bug: cached.mode resolved DataFrame method instead of cached["mode"], yielding empty join','artifacts_preserved':True};write(OUT/'job_ledger.json',old)
 CODE.mkdir();shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n in ('run.py','evaluate.py','thermal_response.py','thermal_metric.py','pipeline.pbs','submit_retry1.py'):shutil.copy2(HERE/n,CODE/n)
 files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};write(CODE/'manifest.json',{'files':files})
 ledger={'study':'mini_temperature1000_linear_fit_retry1','parent_study':'mini_temperature1000_linear_fit','retry_scope':'schema-only cached["mode"] column access; no estimator/data/checkpoint change','snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'cohort':'mixed only; direct full200 IDs40:119; joint=node_mixed+stiffness_mixed; node-only=node_mixed','output_tag':TAG,'v1_failure_jobs':old['jobs'],'jobs':{},'status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat()};lp=OUT/'retry1_job_ledger.json';write(lp,ledger)
 for arm in ARMS:
  j=subprocess.check_output(['qsub','-N','mini_t1000_linear_r1','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm},LSS_TAG={TAG}',str(CODE/'pipeline.pbs')],text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',j);ledger['jobs'][arm]=j;write(lp,ledger)
 ledger['status']='submitted';write(lp,ledger);print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
