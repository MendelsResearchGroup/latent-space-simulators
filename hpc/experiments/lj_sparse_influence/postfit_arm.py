#!/usr/bin/env python3
"""Apply the frozen post-fit physical strain-OLS estimator to one sparse arm."""
import argparse, importlib.util, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(os.environ['LSS_PROJECT_ROOT']); HERE=Path(__file__).resolve().parent; OUT=ROOT/'notebooks/results/lj_sparse_influence'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
run=load('sparse_run',HERE/'run.py')
ols_path=ROOT/'hpc/experiments/lj_latent_influence_edges/evaluate_postfit.py'
assert run.sha(ols_path)=='8fe3196e9e1c826bc5f660e0c3a164c654447fbc33910141de6d06f33566f76b'
ols=load('frozen_ols',ols_path)
def r2(y,p):
 d=np.square(y-y.mean()).sum(); return float(1-np.square(y-p).sum()/d) if d>0 else float('nan')
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args(); output=OUT/'runs'/f'{a.arm}_s{a.seed}'
 assert json.loads((output/'completed.json').read_text())['status']=='completed'
 protocol=json.loads((run.FROZEN/'protocol.json').read_text());data=torch.load(ROOT/protocol['frozen_inputs']['cache'],map_location='cpu',weights_only=False);ae=run.old.load_ae(data);latent=run.old.make_trajectory_propagator('history_attention','mean');latent.load_state_dict(torch.load(ROOT/protocol['frozen_inputs']['latent_propagator'],map_location='cpu',weights_only=False)['state_dict']);latent.eval()
 _, validation, _=run.prepare_records(data,ae,latent,a.arm); checkpoint=torch.load(output/'model.pt',map_location='cpu',weights_only=False);model=run.old.NormalizedAccelerationGNN(hidden_dim=64,steps=4);model.load_state_dict(checkpoint['state_dict']); predicted=run.old.batched_rollout(model,validation,checkpoint['normalizers']); rows=[]
 for rec,pred in zip(validation,predicted):
  sides=run.old.directional_side_indices_from_box(rec['row']['physical_reference'],quantile=.10);truth=rec['row']['physical_positions'];estimate=run.old.physical_positions(rec,pred)
  for h in (10,20,30,40,50,75,100):
   t,e=ols.fitted_response(truth,sides,h),ols.fitted_response(estimate,sides,h);rows.append(dict(arm=a.arm,seed=a.seed,original_index=rec['original_index'],frame=h,true_p_ratio=t['p_ratio'],pred_p_ratio=e['p_ratio'],finite_pair=t['valid'] and e['valid'],true_fit_r2_x=t['fit_r2_x'],true_fit_r2_y=t['fit_r2_y'],pred_fit_r2_x=e['fit_r2_x'],pred_fit_r2_y=e['fit_r2_y'],coordinate_mse=float((estimate[h]-truth[h]).square().mean()),p_ratio_method='frozen physical strain time OLS intercept frames0..horizon fixed sides q0.10 min axial1e-3'))
 table=pd.DataFrame(rows);table.to_csv(output/'postfit_per_network.csv',index=False); summary=[]
 for h,g in table.groupby('frame'):
  v=g[g.finite_pair];summary.append(dict(arm=a.arm,seed=a.seed,frame=h,postfit_p_ratio_r2=r2(v.true_p_ratio.to_numpy(),v.pred_p_ratio.to_numpy()),p_ratio_valid=len(v),p_ratio_total=len(g),coordinate_mse=float(g.coordinate_mse.mean())))
 pd.DataFrame(summary).to_csv(output/'postfit_summary.csv',index=False);(output/'postfit_completed.json').write_text(json.dumps({'status':'completed','estimator_source':'hpc/experiments/lj_latent_influence_edges/evaluate_postfit.py','response_used_for_training_or_selection':False},indent=2)+'\n')
if __name__=='__main__':main()
