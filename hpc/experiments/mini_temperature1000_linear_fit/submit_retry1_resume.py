"""Finish freezing and submit retry1 after its pre-freeze hash helper typo."""
import hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators');OUT=ROOT/'notebooks/results/mini_temperature1000_linear_fit';CODE=OUT/'code_v2';ARMS=('joint_low_uniform_retry1','node_low_uniform_retry1');TAG='retry1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 if not CODE.is_dir() or (CODE/'manifest.json').exists():raise RuntimeError('expected only unfrozen code_v2 staging tree')
 for arm in ARMS:
  for suffix in (f'_{TAG}_smoke',f'_{TAG}'):
   if (OUT/(arm+suffix)).exists():raise FileExistsError(OUT/(arm+suffix))
 files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};write(CODE/'manifest.json',{'files':files})
 old=json.load(open(OUT/'job_ledger.json'))
 ledger={'study':'mini_temperature1000_linear_fit_retry1','parent_study':'mini_temperature1000_linear_fit','retry_scope':'schema-only cached["mode"] column access; no estimator/data/checkpoint change','snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'cohort':'mixed only; direct full200 IDs40:119; joint=node_mixed+stiffness_mixed; node-only=node_mixed','output_tag':TAG,'v1_failure_jobs':old['jobs'],'pre_freeze_incident':'hash-helper local-variable typo; corrected before code_v2 manifest was created or jobs submitted','jobs':{},'status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat()};lp=OUT/'retry1_job_ledger.json';write(lp,ledger)
 for arm in ARMS:
  j=subprocess.check_output(['qsub','-N','mini_t1000_linear_r1','-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm},LSS_TAG={TAG}',str(CODE/'pipeline.pbs')],text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',j);ledger['jobs'][arm]=j;write(lp,ledger)
 ledger['status']='submitted';write(lp,ledger);print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
