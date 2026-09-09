#!/usr/bin/env python3
"""Snapshot and submit the isolated positive-Reid Adam experiment."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'notebooks/results/reid_positive10/adam_v1'
PREPARED=ROOT/'notebooks/results/reid_positive10/study_v1/prepared/reid/prepare/prepared.pt'
COHORT=ROOT/'notebooks/results/reid_positive10/study_v1/cohort.json'
HERE=Path(__file__).resolve().parent
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def qsub(stage,ncpus,mem,walltime,dependency=None):
    env=f'LSS_STAGE={stage},LSS_OUTPUT={OUT},LSS_RUNNER={OUT}/code/run.py'
    cmd=['qsub','-q','mendels_q','-l',f'select=1:ncpus={ncpus}:mem={mem}','-l','place=free:shared','-l',f'walltime={walltime}','-v',env]
    if dependency: cmd += ['-W',f'depend={dependency}']
    cmd += ['-N',f'reid10a_{stage}',str(OUT/'code/run.pbs')]
    return subprocess.check_output(cmd,text=True).strip()
def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');args=p.parse_args()
    if not PREPARED.is_file() or not COHORT.is_file(): raise RuntimeError('frozen study_v1 prepared cohort is required')
    if any((OUT/n).exists() for n in ['code','recipe.json','frozen_designs.json','verification_recipe.json','submission_status.json']): raise RuntimeError('refuse to overwrite an Adam study attempt')
    OUT.mkdir(parents=True,exist_ok=False); code=OUT/'code';code.mkdir()
    for name in ['run.py','run.pbs']: shutil.copy2(HERE/name,code/name)
    deps=[code/'run.py',code/'run.pbs',PREPARED,COHORT,DONOR/'notebooks/network_design/endpoint_physics.py',DONOR/'notebooks/results/network_design/code_endpoint_v1/hashes.json']
    manifest={'study':'reid_positive10/adam_v1','dependencies':{str(x):sha(x) for x in deps},'resources':{'optimize':'4CPU 4GB 60m','verify':'1CPU 2GB 60m'},'submission_order':'optimize ->afterok verify','all_designs_new':True}
    (code/'dependency_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.submit: print(json.dumps({'status':'snapshotted_not_submitted','code':str(code)},indent=2));return
    optimize=qsub('optimize',4,'4gb','01:00:00')
    ledger={'optimize':optimize,'verify':None,'dependency':f'afterok:{optimize}','resources':manifest['resources'],'status':'optimize_submitted_verify_pending'}
    (OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    verify=qsub('verify',1,'2gb','01:00:00',f'afterok:{optimize}')
    ledger.update(verify=verify,status='chain_submitted')
    (OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print((OUT/'submission_status.json').read_text())
if __name__=='__main__':main()
