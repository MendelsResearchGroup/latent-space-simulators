import math,numpy as np,pandas as pd,torch
from graph_utils import calc_p_ratio_rollout_sides,directional_side_indices_from_box
from graph_utils.box import Box
from lss.graph import clone_graph
from lss.dynamics.training import decode_latent_to_graph,latent_step
def phys(g,r):
 o=clone_graph(g).cpu();h=r.reference_box_half_extent.cpu().double()[:2];c=r.reference_box_center.cpu().double()[:2];o.x=o.x.cpu().double();o.x[:,:2]=o.x[:,:2]*h+c;b=o.box;o.box=Box(float(b.x1*h[0]+c[0]),float(b.x2*h[0]+c[0]),float(b.y1*h[1]+c[1]),float(b.y2*h[1]+c[1]),-.1,.1);return o
def evaluate(ae,dyn,stats,data,params,norm,seed,objective,lr,frames):
 rows=[];ae.eval();dyn.eval()
 with torch.no_grad():
  for sim,ztrue,ctx,source,idx in data:
   ref=phys(sim[0],sim[0]);side=directional_side_indices_from_box(ref,quantile=.10);z=ztrue[0]
   for f in range(1,max(frames)+1):
    z=latent_step(dyn,z,stats,loss_mode='delta',context=ctx)
    if f not in frames: continue
    for kind,zz in [('autonomous',z),('reconstruction',ztrue[f])]:
     pred=decode_latent_to_graph(ae,[sim[0]],zz,0,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norm,device='cpu');pp,tp=phys(pred,sim[0]),phys(sim[f],sim[0]);truth=float(calc_p_ratio_rollout_sides([ref,tp],-1,side_idx=side));got=float(calc_p_ratio_rollout_sides([ref,pp],-1,side_idx=side));dt=tp.x[:,:2]-ref.x[:,:2];dp=pp.x[:,:2]-ref.x[:,:2]
     rows.append({'seed':seed,'objective':objective,'lr':lr,'source':source,'split':'validation','original_index':idx,'frame':f,'kind':kind,'true_p_ratio':truth,'pred_p_ratio':got,'finite_pair':math.isfinite(truth) and math.isfinite(got),'displacement_sse':float((dp-dt).square().sum()),'displacement_count':int(dt.shape[0]),'displacement_sum_x':float(dt[:,0].sum()),'displacement_sum_y':float(dt[:,1].sum()),'displacement_square_sum_x':float(dt[:,0].square().sum()),'displacement_square_sum_y':float(dt[:,1].square().sum()),'p_ratio_method':'physical_endpoint_sides_quantile_0.10','initial_frame_only':True})
 return pd.DataFrame(rows)
