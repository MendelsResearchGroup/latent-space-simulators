#!/usr/bin/env python3
"""Snapshot and submit the frozen 12-cell Reid joint-Adam sweep."""
import argparse, hashlib, json, shutil, subprocess
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
OUT=ROOT/'notebooks/results/reid_joint_sweep/study_v1'
PREP=ROOT/'notebooks/results/reid_positive10/study_v1/prepared/reid/prepare/prepared.pt'
COHORT=ROOT/'notebooks/results/reid_positive10/study_v1/cohort.json'
BASE=ROOT/'notebooks/results/reid_positive10/adam_v1'
DONOR=Path('/rg/mendels_prj/alexander.z/DL-course-project')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(stage,output,runner,ncpus,mem,wall,dep=None,name='reidjs'):
    env=f'LSS_STAGE={stage},LSS_OUTPUT={output},LSS_RUNNER={runner}'
    cmd=['qsub','-q','mendels_q','-l',f'select=1:ncpus={ncpus}:mem={mem}','-l','place=free:shared','-l',f'walltime={wall}','-v',env]
    if dep: cmd+=['-W',f'depend={dep}']
    cmd+=['-N',name,str(OUT/'code/run.pbs')]
    return subprocess.check_output(cmd,text=True).strip()
def matrix():
    ids=[int(x['index']) for x in json.loads(COHORT.read_text())['cohort']]
    all_cells=[]
    for lr,bound,pat,mode in product([.005,.02,.08],[.25,1.0],[50,150],['whole_proposal','mask_blocking_nodes']):
        cid=f'lr{lr:g}_b{bound:g}_p{pat}_{mode}'
        all_cells.append(dict(id=cid,learning_rate=lr,log_stiffness_bound=bound,patience=pat,feasibility_mode=mode))
    baseline=next(x for x in all_cells if x['learning_rate']==.02 and x['log_stiffness_bound']==.25 and x['patience']==50 and x['feasibility_mode']=='whole_proposal')
    settings=[]
    for x in all_cells:
        if x['id']==baseline['id']: continue
        out=OUT/'settings'/x['id']; settings.append({**x,'output':str(out.resolve()),'prepared_path':str(PREP.resolve()),'baseline_results_path':str((BASE/'results.csv').resolve()),'global_freeze_completed_path':str((OUT/'global_freeze_completed.json').resolve()),'global_manifest_path':str((OUT/'global_frozen_designs.json').resolve()),'ids':ids})
    return {'ids':ids,'all_cells':all_cells,'baseline':{'id':baseline['id'],'manifest_path':str((BASE/'frozen_designs.json').resolve()),'results_path':str((BASE/'results.csv').resolve())},'settings':settings}
def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    required=[PREP,COHORT,BASE/'frozen_designs.json',BASE/'results.csv']
    if any(not x.is_file() for x in required): raise RuntimeError('prepared cohort and completed baseline Adam artifacts are required')
    if OUT.exists(): raise RuntimeError('refuse to overwrite sweep output')
    OUT.mkdir(parents=True);code=OUT/'code';code.mkdir();
    for name in ['run.py','run.pbs','collect.py']: shutil.copy2(HERE/name,code/name)
    m=matrix(); (OUT/'matrix.json').write_text(json.dumps(m,indent=2)+'\n')
    for s in m['settings']:
        d=Path(s['output']);d.mkdir(parents=True);(d/'config.json').write_text(json.dumps(s,indent=2)+'\n')
    deps=[code/'run.py',code/'run.pbs',code/'collect.py',OUT/'matrix.json',PREP,COHORT,BASE/'frozen_designs.json',BASE/'results.csv',DONOR/'notebooks/network_design/endpoint_physics.py',DONOR/'notebooks/results/network_design/code_endpoint_v1/hashes.json']
    (code/'dependency_manifest.json').write_text(json.dumps({'dependencies':{str(x):sha(x) for x in deps},'matrix_sha256':sha(OUT/'matrix.json'),'resources':{'optimize':'23x 4CPU 2GB 60m','freeze':'1CPU 2GB 10m','verify':'23x 1CPU 2GB 60m','collect':'1CPU 2GB 10m afterany verifiers'}},indent=2)+'\n')
    if not a.submit: print(json.dumps({'status':'snapshotted_not_submitted','output':str(OUT)},indent=2));return
    ledger={'optimizer_jobs':{},'freeze_job':None,'verifier_jobs':{},'collector_job':None,'status':'submitting'};(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    for s in m['settings']:
        jid=call('optimize',s['output'],OUT/'code/run.py',4,'2gb','01:00:00',name='reidjs_opt');ledger['optimizer_jobs'][s['id']]=jid;(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    dep='afterok:'+':'.join(ledger['optimizer_jobs'].values()); freeze=call('freeze',OUT,OUT/'code/collect.py',1,'2gb','00:10:00',dep,'reidjs_freeze');ledger['freeze_job']=freeze;ledger['status']='freeze_submitted';(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    for s in m['settings']:
        jid=call('verify',s['output'],OUT/'code/run.py',1,'2gb','01:00:00',f'afterok:{freeze}','reidjs_verify');ledger['verifier_jobs'][s['id']]=jid;(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    collect_dep='afterany:'+':'.join(ledger['verifier_jobs'].values()); collector=call('collect',OUT,OUT/'code/collect.py',1,'2gb','00:10:00',collect_dep,'reidjs_collect');ledger['collector_job']=collector;ledger['collector_dependency']=collect_dep;(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
    ledger['status']='chain_submitted';(OUT/'submission_status.json').write_text(json.dumps(ledger,indent=2)+'\n')
if __name__=='__main__': main()
