"""Frozen-checkpoint auxiliary boundary-strain linear-fit evaluation; no fitting."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np,pandas as pd,torch
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import encode_frame_latent,decode_latent_to_graph,encode_reference_context,latent_step
from graph_utils import directional_side_indices_from_box
from evaluate import physical,_pref
from thermal_response import fit_response
ROOT=Path(os.environ['LSS_PROJECT_ROOT']);BASE=ROOT/'notebooks/results/all4_coverage_1000';OUT=BASE
ARMS={'all4_one_each':('node_mixed','stiffness_mixed'),'all4_twenty_each':('node_mixed','stiffness_mixed')}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def r2(a,b):
 a=np.asarray(a);b=np.asarray(b);s=((a-a.mean())**2).sum();return float(1-((a-b)**2).sum()/s) if len(a)>=2 and s>0 else float('nan')
def ols(x,y):
 good=np.isfinite(x)&np.isfinite(y)
 if good.sum()!=170:return (float('nan'),float('nan'),float('nan'),int(good.sum()))
 x=x[good];y=y[good]
 if ((x-x.mean())**2).sum()<=0:return (float('nan'),float('nan'),float('nan'),len(x))
 A=np.c_[np.ones(len(x)),x];beta=np.linalg.lstsq(A,y,rcond=None)[0];yp=A@beta
 return float(beta[0]),float(beta[1]),r2(y,yp),len(x)
def endpoint_p(w,h,w0,h0):
 """Historical fixed-side symmetric endpoint estimator, applied to widths/heights."""
 dw=w-w0;dh=h-h0;r1=-dh/(dw+1e-9);r2=-dw/(dh+1e-9)
 return float(r1 if abs(r1)<abs(r2) else r2)
def assert_config():
 if set(ARMS)!= {'all4_one_each','all4_twenty_each'}:raise AssertionError('unexpected checkpoint arms')
 protocol=json.load(open(Path(__file__).with_name('protocol.json')))
 if set(protocol['arms']) != set(ARMS):raise AssertionError('protocol arms differ from frozen checkpoints')
 if any(tuple(protocol['arms'][arm]) != ('node_low','node_mixed','stiffness_low','stiffness_mixed') for arm in ARMS):raise AssertionError('unexpected all-four source order')
 print('linear post-fit static assertions passed: all-four arms, IDs40:119, frames30:199, endpoint199',flush=True)
def sims(result,entry,name):
 spec={'name':name,'label':name,'path':entry['test_path'],'train_count':0,'val_count':80,'split_indices':{'train':[],'val':list(range(40,120)),'test':[]}}
 source={'name':name,'label':name,'path':entry['test_path'],'dataset_name':name,'source_name':name,'coordinate_normalization':'position_normalization','dataset_mixture':[spec]}
 _,data,_,_=resolve_train_val_test(source,result['params'],split_seed=123);assert len(data)==80
 for i,sim in enumerate(data):
  for frame,g in enumerate(sim):g.source_sim_id=i+40;assert g.time==frame*1000
 return data
@torch.no_grad()
def eval_source(result,data,name,arm):
 ae,norm,params=result['ae'],result['normalizers'],result['params'];dyn,stats=result['dyn'],result['latent_stats'];rows=[];ae.eval();dyn.eval()
 for sim in data:
  ref=sim[0];pref,_=_pref(ref);sides=directional_side_indices_from_box(pref,quantile=.10)
  def wh(g):
   q=physical(g,ref);xy=q.x[:,:2].numpy();return float(xy[sides['right'],0].mean()-xy[sides['left'],0].mean()),float(xy[sides['top'],1].mean()-xy[sides['bottom'],1].mean())
  # Fixed frame-zero baseline is the exact saved physical reference, not its
  # normalization inverse (matching the frozen historical endpoint evaluator).
  xy=pref.x[:,:2].numpy();w0=float(xy[sides['right'],0].mean()-xy[sides['left'],0].mean());h0=float(xy[sides['top'],1].mean()-xy[sides['bottom'],1].mean())
  truth=[];recon=[];roll=[];z=encode_frame_latent(ae,sim,0,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norm,device='cpu');ctx=encode_reference_context(ae,sim,pos_dim=2,normalizers=norm,device='cpu',include_temperature=False,include_source_id=False,pool_mode='mean')
  for frame in range(200):
   zr=encode_frame_latent(ae,sim,frame,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norm,device='cpu')
   truth.append((w0,h0) if frame==0 else wh(sim[frame]));recon.append(wh(decode_latent_to_graph(ae,sim,zr,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norm,device='cpu')))
   if frame==0: roll.append((w0,h0))
   else:
    z=latent_step(dyn,z,stats,context=ctx,loss_mode='delta');roll.append(wh(decode_latent_to_graph(ae,sim,z,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norm,device='cpu')))
  for mode,series in [('ae_reconstruction',recon),('autonomous_delta_mlp',roll)]:
   tw,th=np.asarray(truth).T;pw,ph=np.asarray(series).T;tx=(tw-w0)/w0;ty=(th-h0)/h0;px=(pw-w0)/w0;py=(ph-h0)/h0
   ti,ts,tr,tn=ols(tx[30:],ty[30:]);pi,ps,pr,pn=ols(px[30:],py[30:])
   tf=fit_response(tw,th,np.arange(200));pf=fit_response(pw,ph,np.arange(200))
   rows.append({'arm':arm,'source':name,'trajectory_id':ref.source_sim_id,'mode':mode,'linear_true_intercept':ti,'linear_true_slope':ts,'linear_true_p':-ts,'linear_true_fit_r2':tr,'linear_true_pairs':tn,'linear_pred_intercept':pi,'linear_pred_slope':ps,'linear_pred_p':-ps,'linear_pred_fit_r2':pr,'linear_pred_pairs':pn,'endpoint_true_p':endpoint_p(tw[-1],th[-1],w0,h0),'endpoint_pred_p':endpoint_p(pw[-1],ph[-1],w0,h0),'time_ols_true_p':tf['p_ratio'],'time_ols_pred_p':pf['p_ratio'],'time_ols_true_valid_frames':tf['valid_frames'],'time_ols_pred_valid_frames':pf['valid_frames'],'time_ols_true_axial_strain':tf['fitted_axial_total_strain'],'time_ols_pred_axial_strain':pf['fitted_axial_total_strain']})
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--smoke',action='store_true');ap.add_argument('--tag',default='');ap.add_argument('--assert-config',action='store_true');a=ap.parse_args()
 if a.assert_config:assert_config();return
 if a.arm not in ARMS:ap.error('--arm must be a declared frozen checkpoint arm')
 torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']));suffix=(f'_{a.tag}' if a.tag else '')+('_smoke' if a.smoke else '');model_dir=BASE/(a.arm+suffix);folder=model_dir/'auxiliary_linear_fit';folder.mkdir(parents=True,exist_ok=False)
 manifest=json.load(open(BASE/'data_manifest.json'));before={'ae':sha(model_dir/'ae.pt'),'propagator':sha(model_dir/'propagator.pt')};recipe=json.load(open(model_dir/'propagator_recipe.json'));result=load_experiment_bundle(model_dir/'propagator.pt',cfg=recipe['config'],device='cpu');rows=[]
 for name in ARMS[a.arm]:
  data=sims(result,manifest['datasets'][name],name)
  rows+=eval_source(result,data[:1] if a.smoke else data,name,a.arm)
 table=pd.DataFrame(rows);table.to_csv(folder/'per_network.csv',index=False);summ=[]
 for (source,mode),g in table.groupby(['source','mode']):
  q=g.dropna(subset=['linear_true_p','linear_pred_p']);t=g.dropna(subset=['time_ols_true_p','time_ols_pred_p']);summ.append({'source':source,'mode':mode,'linear_p_r2':r2(q.linear_true_p,q.linear_pred_p),'linear_p_mae':float(np.mean(abs(q.linear_true_p-q.linear_pred_p))),'linear_valid':len(q),'linear_total':len(g),'true_fit_r2_p10':float(q.linear_true_fit_r2.quantile(.10)),'true_fit_r2_median':float(q.linear_true_fit_r2.median()),'true_fit_r2_p90':float(q.linear_true_fit_r2.quantile(.90)),'pred_fit_r2_p10':float(q.linear_pred_fit_r2.quantile(.10)),'pred_fit_r2_median':float(q.linear_pred_fit_r2.median()),'pred_fit_r2_p90':float(q.linear_pred_fit_r2.quantile(.90)),'time_ols_p_r2':r2(t.time_ols_true_p,t.time_ols_pred_p),'time_ols_p_mae':float(np.mean(abs(t.time_ols_true_p-t.time_ols_pred_p))),'time_ols_valid':len(t),'time_ols_total':len(g),'time_ols_true_valid_frames_median':float(t.time_ols_true_valid_frames.median()),'time_ols_pred_valid_frames_median':float(t.time_ols_pred_valid_frames.median())})
 pd.DataFrame(summ).to_csv(folder/'summary.csv',index=False)
 # Recompute the historical frame-199 endpoint estimator from this inference
 # and require parity with the frozen evaluator on the identical IDs/cohort.
 parity=[]
 for (source,mode),g in table.groupby(['source','mode']):
  cached=pd.read_csv(model_dir/'test'/source/'per_network.csv')
  cached=cached[(cached['frame']==199)&(cached['mode']==mode)][['trajectory_id','true_p_ratio','pred_p_ratio']]
  joined=g[['trajectory_id','endpoint_true_p','endpoint_pred_p']].merge(cached,on='trajectory_id',how='inner',validate='one_to_one')
  if len(joined)!=len(g):raise AssertionError(f'Endpoint cohort mismatch {source}/{mode}: {len(joined)} != {len(g)}')
  et=float(np.max(np.abs(joined.endpoint_true_p-joined.true_p_ratio)));ep=float(np.max(np.abs(joined.endpoint_pred_p-joined.pred_p_ratio)))
  if et>2e-8 or ep>2e-8:raise AssertionError(f'Endpoint parity mismatch {source}/{mode}: true={et} pred={ep}')
  parity.append({'source':source,'mode':mode,'rows':len(joined),'max_abs_true_p_error':et,'max_abs_pred_p_error':ep})
 pd.DataFrame(parity).to_csv(folder/'endpoint_frame199_parity.csv',index=False)
 after={'ae':sha(model_dir/'ae.pt'),'propagator':sha(model_dir/'propagator.pt')};assert before==after
 endpoint={name:sha(model_dir/'test'/name/'per_network.csv') for name in ARMS[a.arm]}
 (folder/'completed.json').write_text(json.dumps({'status':'completed','arm':a.arm,'tag':a.tag,'smoke':a.smoke,'checkpoint_sha256_before':before,'checkpoint_sha256_after':after,'cached_endpoint_per_network_sha256':endpoint,'data_manifest_sha256':sha(BASE/'data_manifest.json'),'endpoint_frame199_parity':parity,'method':'normalized_boundary_strain OLS frames30..199; p=-slope; all170 finite pairs; fixed physical frame0','postfit_only':True},indent=2)+'\n')
if __name__=='__main__':main()
