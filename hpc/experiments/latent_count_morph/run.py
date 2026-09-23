"""Matched nested training-count study for response-blind latent morphology."""
from __future__ import annotations
import argparse, gc, hashlib, importlib.util, json, os, random, resource, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test, run_latent_experiment, seed_everything
from lss.dynamics.training import encode_frame_latent

ROOT=Path(os.environ['LSS_PROJECT_ROOT']); BASE=ROOT/'notebooks/results/all4_coverage_1000'; OUT=ROOT/'notebooks/results/latent_count_morph'; CODE=Path(__file__).resolve().parent
SOURCES=('node_low','node_mixed','stiffness_low','stiffness_mixed'); LOW=('node_low','stiffness_low'); VAL={'node_low':list(range(20,28)),'node_mixed':list(range(20,28)),'stiffness_low':list(range(20,27)),'stiffness_mixed':list(range(20,27))}; HELD=list(range(40,120)); TRAJECTORY_IDS=(40,50,60,70,80,90,100,110)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def frozen_train_module():
 path=BASE/'code_v5/train.py'; spec=importlib.util.spec_from_file_location('all4_frozen_train',path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
def heldout(result,manifest,name):
 entry=manifest['datasets'][name]; spec=dict(name=name,label=name,path=entry['test_path'],train_count=0,val_count=80,split_indices=dict(train=[],val=HELD,test=[])); source=dict(name=name,label=name,path=entry['test_path'],dataset_name=name,source_name=name,coordinate_normalization='position_normalization',dataset_mixture=[spec]); _,data,_,_=resolve_train_val_test(source,result['params'],split_seed=123); assert len(data)==80; return data
def pcstats(z,p):
 x=z-z.mean(0); _,s,vt=np.linalg.svd(x,full_matrices=False); score=x@vt[0]; eig=s*s; return dict(effective_dimension=float(eig.sum()**2/np.square(eig).sum()),pc1_variance_fraction=float(eig[0]/eig.sum()),pc1_abs_pearson=float(abs(np.corrcoef(score,p)[0,1])))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args(); assert a.count in (1,2,4,8,10,12,14,20); start=time.monotonic();torch.set_num_threads(8); seed_everything(a.seed)
 out=OUT/f'c{a.count}_s{a.seed}';out.mkdir(parents=True,exist_ok=False); manifest=json.loads((BASE/'data_manifest.json').read_text()); ft=frozen_train_module(); reused=a.seed==123 and a.count in (1,20)
 if reused:
  donor=BASE/('all4_one_each_retry_v5_validation30' if a.count==1 else 'all4_twenty_each_retry_v5_validation30'); recipe=json.loads((donor/'propagator_recipe.json').read_text()); result=load_experiment_bundle(donor/'propagator.pt',cfg=recipe['config'],device='cpu'); checkpoint=donor/'ae.pt'
 else:
  cfg=ft.config(out); cfg['model_seed']=a.seed; expected=ft.config(out); expected['model_seed']=a.seed; assert cfg==expected
  source=ft.source_spec(SOURCES,manifest,a.count,False); result=run_latent_experiment(source,cfg,device='cpu')
  exposure=[]
  for split,sims in [('train',result['train_data']),('validation',result['val_data'])]:
   for name in SOURCES:
    ids=list(range(a.count)) if split=='train' else VAL[name]; ss=[s for s in sims if str(s[0].source_name)==name]; assert len(ss)==len(ids)
    for ident,sim in zip(ids,ss): exposure.append(dict(split=split,source=name,trajectory_id=ident,temperature=float(sim[0].temperature),frames=len(sim),first_time=int(sim[0].time),last_time=int(sim[-1].time)))
  pd.DataFrame(exposure).to_csv(out/'exposure.csv',index=False); result['ae_history'].to_csv(out/'ae_warmup_history.csv',index=False); ft.fit_uniform(result,out,60); checkpoint=out/'ae.pt'
 response=pd.read_csv(BASE/'all4_one_each_retry_v5_validation30/test/per_network.csv').query("mode=='ae_reconstruction' and frame==199 and source in @LOW") [['source','trajectory_id','true_p_ratio']].rename(columns={'true_p_ratio':'final_p_ratio'})
 initial=[]; trajectories=[]
 for name in LOW:
  data=heldout(result,manifest,name)
  for ident,sim in zip(HELD,data):
   with torch.no_grad(): z=encode_frame_latent(result['ae'],sim,0,pos_dim=2,node_feature_mode=result['params']['node_feature_mode'],normalizers=result['normalizers'],device='cpu')
   initial.append(dict(source=name,trajectory_id=ident,z0=float(z[0]),z1=float(z[1])))
   if ident in TRAJECTORY_IDS:
    for frame in range(200):
     with torch.no_grad(): z=encode_frame_latent(result['ae'],sim,frame,pos_dim=2,node_feature_mode=result['params']['node_feature_mode'],normalizers=result['normalizers'],device='cpu')
     trajectories.append(dict(source=name,trajectory_id=ident,frame=frame,z0=float(z[0]),z1=float(z[1])))
 initial=pd.DataFrame(initial);initial.to_csv(out/'initial_latents.csv',index=False);pd.DataFrame(trajectories).to_csv(out/'trajectory_latents.csv',index=False);joined=initial.merge(response,on=['source','trajectory_id'],validate='one_to_one'); summary=[]
 for name,g in joined.groupby('source'):summary.append(dict(count=a.count,seed=a.seed,source=name,valid=len(g),total=80,**pcstats(g[['z0','z1']].to_numpy(),g.final_p_ratio.to_numpy())))
 pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
 rec=dict(status='completed',count=a.count,seed=a.seed,reused_existing_checkpoint=reused,training_ids=list(range(a.count)),validation_ids=VAL,heldout_ids=HELD,trajectory_plot_ids=list(TRAJECTORY_IDS),sources=list(SOURCES),plotted_sources=list(LOW),architecture='message_passing_reference16 2D hidden96 tokens32',training='one warmup epoch plus up to60 uniform all-frame epochs; coordinate reconstruction only',selection='same30-trajectory coordinate validation; no response selection',alignment='collector uses centered Frobenius-normalized orthogonal Procrustes on identical heldout initial latents; no p-ratio',response='post-fit low-T endpoint frame199 diagnostic only',checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),data_manifest_sha256=sha(BASE/'data_manifest.json'),script_sha256=sha(__file__),job_id=os.environ.get('PBS_JOBID'),seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (out/'completed.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2),flush=True);del result;gc.collect()
if __name__=='__main__':main()
