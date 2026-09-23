#!/usr/bin/env python3
import hashlib,json,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators');HERE=ROOT/'hpc/experiments/mini_temperature1000_uniform';OUT=ROOT/'notebooks/results/mini_temperature1000_uniform';CODE=OUT/'code_v1';ARMS=('joint_low_uniform','node_low_uniform')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def qsub(script,arm,dep=None):
 c=['qsub','-N','mini_t1000_uniform']
 if dep:c+=['-W','depend='+dep]
 c+=['-v',f'LSS_CODE_ROOT={CODE},LSS_ARM={arm}',str(script)];j=subprocess.check_output(c,text=True).strip();assert re.fullmatch(r'\d+\.[\w.-]+',j);return j
def main():
 if OUT.exists():raise FileExistsError(OUT)
 subprocess.run(['/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python',str(HERE/'prepare_data_manifest.py')],check=True)
 manifest=json.load(open(OUT/'data_manifest.json'));assert set(manifest['datasets'])=={'node_low','node_mixed','stiffness_low','stiffness_mixed'}
 CODE.mkdir();shutil.copytree(ROOT/'src/lss',CODE/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for n in ('train.py','collect.py','evaluate.py','pipeline.pbs','collect.pbs','protocol.json','submit.py'):shutil.copy2(HERE/n,CODE/n)
 shutil.copy2(OUT/'data_manifest.json',CODE/'data_manifest.json');files={str(p.relative_to(CODE)):sha(p) for p in CODE.rglob('*') if p.is_file()};(CODE/'manifest.json').write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
 ledger={'study':'mini_temperature1000_uniform','snapshot':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'data_manifest_sha256':sha(CODE/'data_manifest.json'),'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'split':'direct full200: train0:20,val20:40,test40:120','frames':'0..199000 step1000; 1.99% compression','jobs':{},'status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat()};lp=OUT/'job_ledger.json';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 for a in ARMS:ledger['jobs'][a]={'fit':qsub(CODE/'pipeline.pbs',a)};lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 for a in ARMS:ledger['jobs'][a]['collector']=qsub(CODE/'collect.pbs',a,'afterok:'+ledger['jobs'][a]['fit']);lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
 ledger['status']='submitted';lp.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n');print(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
