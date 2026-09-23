#!/usr/bin/env python3
import hashlib,json,os,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators');HERE=ROOT/'hpc/experiments/all4_coverage_1000';OUT=ROOT/'notebooks/results/all4_coverage_1000';CODE=OUT/'code_v2';ARMS=('all4_one_each','all4_twenty_each')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def qsub(script,arm,dep=None):
 c=['qsub','-N','all4cov1000']
 if dep:c+=['-W','depend='+dep]
 c+=['-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(script)];j=subprocess.check_output(c,text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',j);return j
def main():
 env=dict(os.environ,LSS_PROJECT_ROOT=str(ROOT),PYTHONPATH=f'{ROOT}/src:/rg/mendels_prj/alexander.z/MetaForge/src',OMP_NUM_THREADS='1')
 subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/'train.py'),'--assert-config'],check=True,env=env)
 if not OUT.exists():
  subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/'prepare_data_manifest.py')],check=True)
  manifest=json.load(open(OUT/'data_manifest.json'));assert set(manifest['datasets'])=={'node_low','node_mixed','stiffness_low','stiffness_mixed'}
  CODE.mkdir();shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  for n in ('train.py','collect.py','evaluate.py','linear_run.py','thermal_response.py','thermal_metric.py','pipeline.pbs','collect.pbs','protocol.json','submit.py','prepare_data_manifest.py'):shutil.copy2(HERE/n,CODE/n)
  shutil.copy2(OUT/'data_manifest.json',CODE/'data_manifest.json');files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};(CODE/'manifest.json').write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
  ledger={'study':'all4_coverage_1000','snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'data_manifest_sha256':sha(CODE/'data_manifest.json'),'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'split':{'all4_one_each':{'train':'IDs0','validation':'IDs20:39','heldout_development':'IDs40:119'},'all4_twenty_each':{'train':'IDs0:19','validation':'IDs20:39','heldout_development':'IDs40:119'}},'frames':'0..199000 step1000; 1.99% compression','endpoint':'physical fixed-frame0 q0.10 at frame199','auxiliary_postfit':'normalized boundary-strain OLS frames30..199 with strict endpoint parity on node_mixed and stiffness_mixed only; not fitting/selection','mixed_temperature_qualification':'all4_one_each mixed ID0 is one trajectory only; no mixed-temperature-distribution coverage claim','jobs':{},'status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat()};lp=OUT/'job_ledger.json';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 elif not CODE.exists():
  lp=OUT/'job_ledger.json';ledger=json.load(open(lp));assert ledger['status']=='submitting' and not ledger['jobs'], 'resume only the recorded no-job submission failure'
  assert (OUT/'code_v1'/'manifest.json').is_file() and sha(OUT/'data_manifest.json')==ledger['data_manifest_sha256']
  CODE.mkdir();shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  for n in ('train.py','collect.py','evaluate.py','linear_run.py','thermal_response.py','thermal_metric.py','pipeline.pbs','collect.pbs','protocol.json','submit.py','prepare_data_manifest.py'):shutil.copy2(HERE/n,CODE/n)
  shutil.copy2(OUT/'data_manifest.json',CODE/'data_manifest.json');files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};(CODE/'manifest.json').write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
  ledger['submission_attempt_failure']={'reason':'sandbox qsub could not create /var/tmp PBS script','preserved_snapshot':'notebooks/results/all4_coverage_1000/code_v1'};ledger['snapshot']=str(CODE.relative_to(ROOT));ledger['code_manifest_sha256']=sha(CODE/'manifest.json');ledger['paired_topology_qualification']='low/mixed same-ID records are paired topology/source identities; one_each has 4 trajectories but 2 unique structures, twenty_each has 80 trajectories but 40 unique structures; comparison changes both structure and thermal-trajectory coverage.';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 else:
  lp=OUT/'job_ledger.json';ledger=json.load(open(lp));assert ledger['status']=='submitting' and not ledger['jobs'];assert sha(CODE/'manifest.json')==ledger['code_manifest_sha256']
 for a in ARMS:ledger['jobs'][a]={'fit':qsub(CODE/'pipeline.pbs',a)};lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 for a in ARMS:ledger['jobs'][a]['collector']=qsub(CODE/'collect.pbs',a,'afterok:'+ledger['jobs'][a]['fit']);lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 ledger['status']='submitted';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n');print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
