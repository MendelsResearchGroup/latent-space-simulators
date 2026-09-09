#!/usr/bin/env python3
"""Snapshot and submit the frozen positive-Reid prepare/optimize/verify chain."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'notebooks/results/reid_positive10/study_v1'
HERE=Path(__file__).resolve().parent
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def qsub(stage, ncpus, mem, walltime, dependency=None):
    env=f'LSS_STAGE={stage},LSS_OUTPUT={OUT},LSS_RUNNER={OUT}/code/{"prepare.py" if stage=="prepare" else "run.py"}'
    cmd=['qsub','-q','mendels_q','-l',f'select=1:ncpus={ncpus}:mem={mem}','-l','place=free:shared','-l',f'walltime={walltime}','-v',env]
    if dependency: cmd += ['-W',f'depend={dependency}']
    cmd += ['-N',f'reid10_{stage}',str(OUT/'code/run.pbs')]
    return subprocess.check_output(cmd,text=True).strip()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--submit',action='store_true'); args=parser.parse_args()
    if not (OUT/'cohort.json').is_file(): raise RuntimeError('root-owned cohort.json is required')
    if any((OUT/n).exists() for n in ['prepared','recipe.json','frozen_designs.json','verification_recipe.json']): raise RuntimeError('refuse to overwrite an existing study attempt')
    code=OUT/'code'; code.mkdir(exist_ok=False)
    for name in ['run.py','prepare.py','run.pbs']:
        shutil.copy2(HERE/name,code/name)
    shutil.copy2(OUT/'cohort.json',code/'cohort.json')
    deps=[code/'run.py',code/'prepare.py',code/'run.pbs',code/'cohort.json',DONOR/'notebooks/results/network_design/calibrated_direction_v1/reid/prepare/prepared.pt',DONOR/'notebooks/network_design/endpoint_physics.py',DONOR/'notebooks/results/network_design/code_endpoint_v1/hashes.json']
    manifest={'study':'reid_positive10/study_v1','cohort_sha256':sha(OUT/'cohort.json'),'dependencies':{str(p):sha(p) for p in deps},'resources':{'prepare':'4CPU 2GB 20m','optimize':'4CPU 4GB 60m','verify':'1CPU 2GB 60m'},'submission_order':'prepare ->afterok optimize ->afterok verify'}
    (code/'dependency_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not args.submit:
        print(json.dumps({'status':'snapshotted_not_submitted','code':str(code),'manifest':str(code/'dependency_manifest.json')},indent=2)); return
    ledger={'resources':manifest['resources'],'dependencies':{},'status':'submitting'}
    def save(): (OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    save()
    prepare=qsub('prepare',4,'2gb','00:20:00');ledger['prepare']=prepare;save()
    optimize=qsub('optimize',4,'4gb','01:00:00',f'afterok:{prepare}');ledger['optimize']=optimize;ledger['dependencies']['optimize']=f'afterok:{prepare}';save()
    verify=qsub('verify',1,'2gb','01:00:00',f'afterok:{optimize}');ledger['verify']=verify;ledger['dependencies']['verify']=f'afterok:{optimize}';ledger['status']='submitted';save()
    (OUT/'submission_status.json').write_text(json.dumps({'prepare':prepare,'optimize':optimize,'verify':verify,'dependencies':{'optimize':f'afterok:{prepare}','verify':f'afterok:{optimize}'},'resources':manifest['resources']},indent=2)+'\n')
    print(json.dumps(json.loads((OUT/'submission_status.json').read_text()),indent=2))
if __name__=='__main__': main()
