"""Frozen-checkpoint prefix affine-strain evaluation; no fitting or selection."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import encode_frame_latent, decode_latent_to_graph, encode_reference_context, latent_step
from graph_utils import directional_side_indices_from_box
from evaluate import physical, _pref

ROOT=Path(os.environ['LSS_PROJECT_ROOT']); BASE=ROOT/'notebooks/results/mini_temperature1000_mixed_only'; OLD=BASE/'linear_fit'; OUT=BASE/'prefix_linear_fit'
ARMS={'joint_mixed_only':('node_mixed','stiffness_mixed')}; HORIZONS=tuple(range(37,200))
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def r2(y,p):
 y=np.asarray(y);p=np.asarray(p);sst=((y-y.mean())**2).sum()
 return float(1-((y-p)**2).sum()/sst) if len(y)>=2 and sst>0 else float('nan')
def ols(x,y):
 x=np.asarray(x);y=np.asarray(y);good=np.isfinite(x)&np.isfinite(y)
 if len(x)<8 or good.sum()!=len(x):return (np.nan,np.nan,np.nan,int(good.sum()))
 if ((x-x.mean())**2).sum()<=0:return (np.nan,np.nan,np.nan,len(x))
 beta=np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0];return float(beta[0]),float(beta[1]),r2(y,np.c_[np.ones(len(x)),x]@beta),len(x)
def endpoint_p(w,h,w0,h0):
 dw=w-w0;dh=h-h0;r1=-dh/(dw+1e-9);r2=-dw/(dh+1e-9);return float(r1 if abs(r1)<abs(r2) else r2)
def assert_config():
 manifest=json.load(open(BASE/'data_manifest.json'));assert set(ARMS)=={'joint_mixed_only'};assert set(manifest['datasets'])=={'node_low','node_mixed','stiffness_low','stiffness_mixed'}
 for arm,sources in ARMS.items():
  assert (BASE/arm/'ae.pt').is_file() and (BASE/arm/'propagator.pt').is_file()
  assert (OLD/arm/'per_network.csv').is_file() and (OLD/arm/'summary.csv').is_file()
  for source in sources: assert Path(manifest['datasets'][source]['test_path']).is_file() and (BASE/arm/'test'/source/'per_network.csv').is_file()
 print('prefix static assertions passed: frozen mixed-only IDs40:119, horizons37..199, parity guards',flush=True)
def load_sims(result,entry,name):
 spec={'name':name,'label':name,'path':entry['test_path'],'train_count':0,'val_count':80,'split_indices':{'train':[],'val':list(range(40,120)),'test':[]}}
 source={'name':name,'label':name,'path':entry['test_path'],'dataset_name':name,'source_name':name,'coordinate_normalization':'position_normalization','dataset_mixture':[spec]}
 _,data,_,_=resolve_train_val_test(source,result['params'],split_seed=123);assert len(data)==80
 for i,sim in enumerate(data):
  for frame,g in enumerate(sim):g.source_sim_id=i+40;assert g.time==frame*1000
 return data
@torch.no_grad()
def evaluate_source(result,data,source,arm):
 ae,norm,params=result['ae'],result['normalizers'],result['params'];dyn,stats=result['dyn'],result['latent_stats'];ae.eval();dyn.eval();rows=[]
 for sim in data:
  ref=sim[0];pref,_=_pref(ref);sides=directional_side_indices_from_box(pref,quantile=.10);xy=pref.x[:,:2].numpy();w0=float(xy[sides['right'],0].mean()-xy[sides['left'],0].mean());h0=float(xy[sides['top'],1].mean()-xy[sides['bottom'],1].mean())
  def wh(graph):
   xy=physical(graph,ref).x[:,:2].numpy();return float(xy[sides['right'],0].mean()-xy[sides['left'],0].mean()),float(xy[sides['top'],1].mean()-xy[sides['bottom'],1].mean())
  truth=[];recon=[];roll=[];z=encode_frame_latent(ae,sim,0,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norm,device='cpu');ctx=encode_reference_context(ae,sim,pos_dim=2,normalizers=norm,device='cpu',include_temperature=False,include_source_id=False,pool_mode='mean')
  for frame in range(200):
   zr=encode_frame_latent(ae,sim,frame,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norm,device='cpu');truth.append((w0,h0) if frame==0 else wh(sim[frame]));recon.append(wh(decode_latent_to_graph(ae,sim,zr,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norm,device='cpu')))
   if frame==0:roll.append((w0,h0))
   else:z=latent_step(dyn,z,stats,context=ctx,loss_mode='delta');roll.append(wh(decode_latent_to_graph(ae,sim,z,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norm,device='cpu')))
  tw,th=np.asarray(truth).T;tx=(tw-w0)/w0;ty=(th-h0)/h0
  for mode,series in [('ae_reconstruction',recon),('autonomous_delta_mlp',roll)]:
   pw,ph=np.asarray(series).T;px=(pw-w0)/w0;py=(ph-h0)/h0
   for horizon in HORIZONS:
    ti,ts,tr,tn=ols(tx[30:horizon+1],ty[30:horizon+1]);pi,ps,pr,pn=ols(px[30:horizon+1],py[30:horizon+1]);rows.append({'arm':arm,'source':source,'trajectory_id':ref.source_sim_id,'mode':mode,'horizon':horizon,'frames_start':30,'frames_end':horizon,'expected_pairs':horizon-29,'linear_true_intercept':ti,'linear_true_slope':ts,'linear_true_p':-ts,'linear_true_fit_r2':tr,'linear_true_pairs':tn,'linear_pred_intercept':pi,'linear_pred_slope':ps,'linear_pred_p':-ps,'linear_pred_fit_r2':pr,'linear_pred_pairs':pn,'endpoint_true_p':endpoint_p(tw[-1],th[-1],w0,h0),'endpoint_pred_p':endpoint_p(pw[-1],ph[-1],w0,h0)})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--smoke',action='store_true');ap.add_argument('--assert-config',action='store_true');a=ap.parse_args()
 if a.assert_config:assert_config();return
 if a.arm not in ARMS:ap.error('declared --arm required')
 torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']));folder=OUT/(a.arm+('_smoke' if a.smoke else ''));folder.mkdir(parents=True,exist_ok=False);manifest=json.load(open(BASE/'data_manifest.json'));model=BASE/a.arm;before={'ae':sha(model/'ae.pt'),'propagator':sha(model/'propagator.pt')};recipe=json.load(open(model/'propagator_recipe.json'));result=load_experiment_bundle(model/'propagator.pt',cfg=recipe['config'],device='cpu');rows=[]
 for source in ARMS[a.arm]:
  data=load_sims(result,manifest['datasets'][source],source);rows+=evaluate_source(result,data[:1] if a.smoke else data,source,a.arm)
 table=pd.DataFrame(rows);assert table.groupby(['source','mode','trajectory_id'])['horizon'].nunique().eq(len(HORIZONS)).all();table.to_csv(folder/'per_network_prefix.csv',index=False)
 summary=[]
 for (source,mode,horizon),g in table.groupby(['source','mode','horizon']):
  q=g.dropna(subset=['linear_true_p','linear_pred_p']);summary.append({'source':source,'mode':mode,'horizon':horizon,'linear_p_r2':r2(q.linear_true_p,q.linear_pred_p),'linear_p_mae':float(np.mean(abs(q.linear_true_p-q.linear_pred_p))),'linear_valid':len(q),'linear_total':len(g),'true_fit_r2_median':float(q.linear_true_fit_r2.median()),'pred_fit_r2_median':float(q.linear_pred_fit_r2.median())})
 summary=pd.DataFrame(summary);summary.to_csv(folder/'summary_prefix.csv',index=False)
 parity=[]
 for (source,mode),g in table[table.horizon.eq(199)].groupby(['source','mode']):
  old=pd.read_csv(OLD/a.arm/'per_network.csv');old=old[(old.source==source)&(old['mode']==mode)][['trajectory_id','linear_true_p','linear_pred_p']]
  joined=g[['trajectory_id','linear_true_p','linear_pred_p','endpoint_true_p','endpoint_pred_p']].merge(old,on='trajectory_id',suffixes=('_new','_old'),validate='one_to_one');assert len(joined)==len(g)
  cached=pd.read_csv(model/'test'/source/'per_network.csv');cached=cached[(cached['frame']==199)&(cached['mode']==mode)][['trajectory_id','true_p_ratio','pred_p_ratio']];joined=joined.merge(cached,on='trajectory_id',validate='one_to_one')
  errors={'linear_true':float(np.abs(joined.linear_true_p_new-joined.linear_true_p_old).max()),'linear_pred':float(np.abs(joined.linear_pred_p_new-joined.linear_pred_p_old).max()),'endpoint_true':float(np.abs(joined.endpoint_true_p-joined.true_p_ratio).max()),'endpoint_pred':float(np.abs(joined.endpoint_pred_p-joined.pred_p_ratio).max())}
  if max(errors.values())>2e-8:raise AssertionError(f'parity {source}/{mode}: {errors}')
  parity.append({'source':source,'mode':mode,'rows':len(joined),**errors})
 pd.DataFrame(parity).to_csv(folder/'parity_t199.csv',index=False);after={'ae':sha(model/'ae.pt'),'propagator':sha(model/'propagator.pt')};assert before==after
 (folder/'completed.json').write_text(json.dumps({'status':'completed','arm':a.arm,'smoke':a.smoke,'checkpoint_sha256_before':before,'checkpoint_sha256_after':after,'endpoint_and_linear_t199_parity':parity,'method':'prefix affine normalized boundary strain OLS frames30..horizon, horizon37..199; p=-slope; all pairs finite; postfit only'},indent=2)+'\n')
if __name__=='__main__':main()
