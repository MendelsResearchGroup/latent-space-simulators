"""AE-only dynamics-data latent-capacity study; mixed-T is evaluation-only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['LSS_PROJECT_ROOT'])
CODE=Path(os.environ['LSS_CODE_ROOT'])
RESULTS_ROOT=Path(os.environ.get('LSS_RESULTS_ROOT',ROOT/'notebooks/results'))
def load_portable_manifest(path):
    manifest=json.loads(path.read_text())
    for spec in manifest.get('sources',{}).values():spec['path']=str(ROOT/'data'/Path(spec['path']).name)
    return manifest
sys.path.insert(0,str(CODE/'src'))
import pandas as pd
import torch
from lss.dynamics.experiment import run_latent_experiment,seed_everything,resolve_train_val_test
from lss.dynamics.training import encode_frame_latent,decode_latent_to_graph
from latent_diagnostic_metrics import metrics,summarize

VARIANTS=('near08','dim2','dim4','dim8','width128','tokens32','frames101','source_mean')


def build_recipe(baseline,variant,output):
    source=copy.deepcopy(baseline['source']);cfg=copy.deepcopy(baseline['config'])
    mixed=copy.deepcopy(baseline['mixed_evaluation'])
    for container in (source,mixed):
        for spec in container.get('dataset_mixture',[]):
            spec['path']=str(ROOT/'data'/Path(spec['path']).name)
        if container.get('path'):
            container['path']=str(ROOT/'data'/Path(container['path']).name)
    manifest=load_portable_manifest(Path(os.environ.get('LSS_SPLIT_MANIFEST',RESULTS_ROOT/'lj_ae_08_bridge/split_manifest.json')))
    for spec in source['dataset_mixture']:
        spec['split_indices']['train']=manifest['sources'][spec['name']]['split_indices']['train']
        spec['train_count']=len(spec['split_indices']['train'])
    for spec in [*source['dataset_mixture'],*mixed['dataset_mixture']]:
        spec.update(append_lj_indicator=True,lj_max_graph_distance=3 if spec['name']=='lj_noisy' else None)
    cfg.update(should_train_propagator=False,should_rollout=False,cache_path=str(output/'ae.pt'),force_train=True)
    cfg.pop('propagator_config')
    ae=cfg['ae_config']
    ae.update(latent_dim=6,hidden_size=192,latent_tokens=46,edge_feature_dim=5,
        max_train_frames_per_sim=150,max_val_frames_per_sim=150,max_epochs=50,patience=8,
        gradient_method='mean',checkpoint_metric=None,checkpoint_mode='min',pratio_eval_every=0)
    if variant.startswith('dim'):ae['latent_dim']=int(variant[3:])
    if variant=='width128':ae['hidden_size']=128
    if variant=='tokens32':ae['latent_tokens']=32
    if variant=='frames101':ae.update(max_train_frames_per_sim=101,max_val_frames_per_sim=101)
    if variant=='source_mean':ae['gradient_method']='source_mean'
    return source,cfg,mixed


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    baseline_path=Path(os.environ.get('LSS_HISTORICAL_BASELINE_ROOT',RESULTS_ROOT))/f'06b_ae_mixed_ablation/exclude_mixed_d2_s{args.seed}/recipe.json'
    baseline=json.loads(baseline_path.read_text())
    output=RESULTS_ROOT/f'lj_ae_08_bridge/{args.variant}_s{args.seed}_{os.environ["LSS_CODE_VERSION"]}'
    source,cfg,mixed_source=build_recipe(baseline,args.variant,output)
    cfg["model_seed"]=args.seed
    assert {s['name'] for s in source['dataset_mixture']}=={'reid','depablo_low_temp','lj_noisy'}
    manifest=load_portable_manifest(Path(os.environ.get('LSS_SPLIT_MANIFEST',RESULTS_ROOT/'lj_ae_08_bridge/split_manifest.json')))
    for spec in [*source['dataset_mixture'],*mixed_source['dataset_mixture']]:
        digest=hashlib.sha256()
        with Path(spec['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=manifest['sources'][spec['name']]['sha256']:raise ValueError('Dataset changed')
    output.mkdir(parents=True,exist_ok=False)
    (output/'recipe.json').write_text(json.dumps(dict(source=source,config=cfg,mixed_evaluation=mixed_source,
        variant=args.variant,seed=args.seed,job_id=os.environ['PBS_JOBID'],code_root=str(CODE),
        baseline_recipe=str(baseline_path),baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        response_selection='response' in args.variant,
        design='08-inspired AE-only bridge. Train30/30/60 Reid/lowT/LJ,20val each; user-authorized reserved LJ migration with topology exclusions. 6D/192hidden/46tokens/150frames/mean objective anchor; ordinary reconstruction selection. Historical frozen edge convention retained to isolate recipe; not an exact08 replay.',
        contrast=args.variant),indent=2))
    seed_everything(args.seed);started=time.time()
    result=run_latent_experiment(source,cfg,device='cpu')
    assert not result['test_data']
    assert all(s[0].source_name!='depablo_mixed_temp' for k in ('train_data','val_data') for s in result[k])
    result['ae_history'].to_csv(output/'ae_history.csv',index=False)
    train,mixed,test,_=resolve_train_val_test(mixed_source,result['params'],split_seed=cfg['split_seed'])
    assert not train and not test and len(mixed)==20
    ae=result['ae'];ae.eval();params=result['params'];norms=result['normalizers']
    counts={};rows=[]
    with torch.no_grad():
        for sim in [*result['val_data'],*mixed]:
            name=sim[0].source_name;i=counts.get(name,0);counts[name]=i+1
            identity=manifest['sources'][name]['split_indices']['val'][i]
            for frame in (5,10,25,50,75,100,125,149,150):
                z=encode_frame_latent(ae,sim,frame,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norms,device='cpu')
                pred=decode_latent_to_graph(ae,sim,z,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norms,device='cpu')
                rows.append(dict(source=name,original_index=identity,frame=frame,**metrics(sim,frame,pred)))
            print('evaluated',name,identity,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'validation_rows.csv',index=False)
    summary=summarize(frame,['source','frame']);summary.to_csv(output/'source_summary.csv',index=False)
    print(summary.query('frame==100').to_string(index=False),flush=True)
    (output/'completed.json').write_text(json.dumps(dict(seconds=time.time()-started,seed=args.seed,variant=args.variant,
        job_id=os.environ['PBS_JOBID'],split='validation',mixed_evaluation_only=True)))


if __name__=='__main__':main()
