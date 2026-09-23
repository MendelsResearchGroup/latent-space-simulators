"""Frozen validation-only tests of reference offsets and boundary motion."""
import json, os, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lss.dynamics.experiment import _load_ae_cache
from lss.dynamics.training import encode_frame_latent, decode_latent_to_graph
from lss.graph import clone_graph
from graph_utils import directional_side_indices_from_box, calc_p_ratio_rollout_sides
from evaluate import physical, _pref
ROOT=Path(os.environ['LSS_PROJECT_ROOT']);OUT=ROOT/'notebooks/results/mini_ae_audit'
FRAMES=[0,1,2,3,4,5,10,19]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spans(x,s):return torch.stack((x[s['right'],0].mean()-x[s['left'],0].mean(),x[s['top'],1].mean()-x[s['bottom'],1].mean()))
def r2(a,b):
    valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
    denom=((a-a.mean())**2).sum()
    return float(1-((a-b)**2).sum()/denom) if len(a)>1 and denom>0 else float('nan'),int(valid.sum())
rows=[];provenance=[]
for study in ['mini_stiffness','mini_node']:
    folder=ROOT/'notebooks/results'/study/'runs/d2_s123'
    recipe=json.loads((folder/'recipe.json').read_text())
    result=_load_ae_cache(folder/'ae.pt',recipe['config'],device='cpu')
    ae=result['ae'];norm=result['normalizers'];params=result['params']
    before=sha(folder/'ae.pt')
    assert len(result['val_data'])==20
    with torch.no_grad():
        for sim in result['val_data']:
            ref,inv=_pref(sim[0]);sides=directional_side_indices_from_box(ref,quantile=.10)
            xy=ref.x[:,:2].double();s0=spans(xy,sides)
            predictions={}
            for frame in FRAMES:
                z=encode_frame_latent(ae,sim,frame,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norm,device='cpu')
                decoded=decode_latent_to_graph(ae,sim,z,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norm,device='cpu')
                predictions[frame]=physical(decoded,sim[0]).x[:,:2].double()
            offset=predictions[0]-xy
            for frame in FRAMES:
                truth=xy if frame==0 else physical(sim[frame],sim[0]).x[:,:2].double()
                tg=clone_graph(ref);tg.x[:,:2]=truth
                truep=float(calc_p_ratio_rollout_sides([ref,tg],side_idx=sides))
                truechange=spans(truth,sides)-s0
                original_error=predictions[frame]-truth
                cosine=float((original_error*offset).sum()/(original_error.norm()*offset.norm()))
                variants={'original':predictions[frame], 'reference_subtracted':predictions[frame]-offset,
                          'translation_removed':predictions[frame]-offset.mean(0)}
                for mode,pred in variants.items():
                    pg=clone_graph(ref);pg.x[:,:2]=pred
                    predp=float(calc_p_ratio_rollout_sides([ref,pg],side_idx=sides))
                    change=spans(pred,sides)-s0;error=pred-truth
                    boundary=np.unique(np.concatenate(list(sides.values())))
                    rows.append(dict(study=study,split='validation',seed=123,trajectory_id=int(sim[0].source_sim_id),frame=frame,mode=mode,
                        true_p_ratio=truep,pred_p_ratio=predp,sse=float(error.square().sum()),components=error.numel(),
                        boundary_sse=float(error[boundary].square().sum()),boundary_components=2*len(boundary),
                        true_displacement_sse=float((truth-xy).square().sum()),
                        true_dx=float(truechange[0]),true_dy=float(truechange[1]),pred_dx=float(change[0]),pred_dy=float(change[1]),
                        true_x_dominant=bool(truechange[0].abs()>=truechange[1].abs()),pred_x_dominant=bool(change[0].abs()>=change[1].abs()),
                        original_error_reference_cosine=cosine,physical_inverse_maxabs=inv))
                if frame==0:torch.testing.assert_close(variants['reference_subtracted'],xy,rtol=0,atol=1e-14)
                if np.isfinite(predp) and np.isfinite(truep):
                    # Translation control must preserve directional boundary widths/heights.
                    torch.testing.assert_close(spans(variants['translation_removed'],sides),spans(predictions[frame],sides),rtol=1e-10,atol=1e-12)
    assert sha(folder/'ae.pt')==before
    provenance.append(dict(study=study,ae_sha256=before,recipe_sha256=sha(folder/'recipe.json'),split_indices=recipe['source']['dataset_mixture'][0]['split_indices']['val']))
    del result,ae
    print(study,'complete',flush=True)
table=pd.DataFrame(rows);table.to_csv(OUT/'per_network.csv',index=False)
summary=[]
for (study,frame,mode),g in table.groupby(['study','frame','mode']):
    score,count=r2(g.true_p_ratio.to_numpy(),g.pred_p_ratio.to_numpy())
    finite=g[np.isfinite(g.true_p_ratio)&np.isfinite(g.pred_p_ratio)]
    summary.append(dict(study=study,frame=frame,mode=mode,p_ratio_r2=score,valid=count,total=len(g),
        mse=g.sse.sum()/g.components.sum(),boundary_mse=g.boundary_sse.sum()/g.boundary_components.sum(),
        true_motion_rms=np.sqrt(g.true_displacement_sse.sum()/g.components.sum()),
        p_bias=(finite.pred_p_ratio-finite.true_p_ratio).mean(),
        true_dx_rms=np.sqrt(np.mean(g.true_dx**2)),true_dy_rms=np.sqrt(np.mean(g.true_dy**2)),
        dx_rmse=np.sqrt(np.mean((g.pred_dx-g.true_dx)**2)),dy_rmse=np.sqrt(np.mean((g.pred_dy-g.true_dy)**2)),
        direction_switches=int((g.true_x_dominant!=g.pred_x_dominant).sum()),
        median_error_reference_cosine=g.original_error_reference_cosine.median()))
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
(OUT/'completed.json').write_text(json.dumps(dict(job_id=os.environ['PBS_JOBID'],status='completed',provenance=provenance,frames=FRAMES,rows=len(table),final_test_used=False,training=False),indent=2)+'\n')
