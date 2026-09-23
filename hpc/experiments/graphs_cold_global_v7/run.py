"""Validation-only short-horizon initial-motion OLS-residual box response."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; V4=ROOT/"hpc/experiments/graphs_cold_global_v4/run.py"
spec=importlib.util.spec_from_file_location("v4",V4); v4=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(v4); base=v4.base
from initial_motion_response import InitialMotionBoxResponse
DT=200.
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def boundary(split,ids):
    if set(ids)-set(split["validation"]): raise RuntimeError("Validation-only harness must never load final_test.")
def batches(ids,rng):
    x=list(ids); rng.shuffle(x); return [x[i:i+8] for i in range(0,len(x),8)]
def node_graph(frames,cutoff):
    f0,f1=frames[0],frames[1]; frac0=base.fractional_coordinates(f0["pos"],f0["box_tensor"]); frac1=base.fractional_coordinates(f1["pos"],f1["box_tensor"]); velocity=base.unit_minimum_image(frac1-frac0)/(f1["time"]-f0["time"]); edge,index=base.raw_edge(f0,cutoff); return torch.cat((frac0,velocity),1),f0["atom_types"]-1,edge,index
def target_x(frames,h):
    delta=(frames[1]["box_tensor"][0]-frames[0]["box_tensor"][0])/(frames[1]["time"]-frames[0]["time"]); return frames[0]["box_tensor"][0]+h*DT*delta
def contexts(frames,hs):
    velocity=(frames[1]["box_tensor"]-frames[0]["box_tensor"])/(frames[1]["time"]-frames[0]["time"])
    return torch.stack([torch.cat((frames[0]["box_tensor"],target_x(frames,int(h)).reshape(1),velocity)) for h in hs])
def targets(frames,hs): return torch.stack([frames[int(h)]["box_tensor"][1:]-frames[0]["box_tensor"][1:] for h in hs])
def fit(train):
    hs=torch.arange(3,101); c=torch.cat([contexts(f,hs) for f in train]); y=torch.cat([targets(f,hs) for f in train]); mean,std=c.mean(0),c.std(0).clamp_min(1e-12); x=torch.cat((torch.ones((len(c),1)),(c-mean)/std),1); coef=torch.linalg.lstsq(x,y).solution; r=y-x@coef
    return {"context_mean":mean,"context_std":std,"ols_coefficients":coef,"residual_mean":r.mean(0),"residual_std":r.std(0).clamp_min(1e-12)}
def predict(model,graph,frames,hs,norm,beta,embedding=None):
    c=contexts(frames,hs); z=(c-norm["context_mean"])/norm["context_std"]; x=torch.cat((torch.ones((len(c),1)),z),1); ols=x@norm["ols_coefficients"]; e=model.encode(*graph) if embedding is None else embedding; residual=model.decode(e,z)*norm["residual_std"]+norm["residual_mean"]; initial=frames[0]["box_tensor"][1:]; return F.softplus(beta*(initial+ols+residual))/beta+1e-6,F.softplus(beta*(initial+ols))/beta+1e-6
def affine(frames,h,yz):
    box=torch.cat((target_x(frames,h).reshape(1),yz)); return base.fractional_coordinates(frames[0]["pos"],frames[0]["box_tensor"])*box,box
def metrics(rows):
    yr,yv=base.r2(rows,"true_p_y_over_x","pred_p_y_over_x"); zr,zv=base.r2(rows,"true_p_z_over_x","pred_p_z_over_x"); return {"position_mse_physical_minimum_image":float(np.mean([r["position_mse_physical_minimum_image"] for r in rows])),"yz_box_length_mse":float(np.mean([r["yz_box_length_mse"] for r in rows])),"p_ratio_r2_y_over_x":yr,"p_ratio_r2_z_over_x":zr,"p_ratio_y_valid_total":[yv,len(rows)],"p_ratio_z_valid_total":[zv,len(rows)]}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,required=True); p.add_argument("--smoke",action="store_true"); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); protocol=json.loads((HERE/"protocol.json").read_text()); split_path=(HERE/protocol["split_manifest"]).resolve(); split=json.loads(split_path.read_text()); ids=[i for k in ("train","validation","final_test") for i in split[k]]
 if len(split["train"])!=48 or len(split["validation"])!=12 or len(split["final_test"])!=12 or len(set(ids))!=72 or a.seed!=123: raise ValueError("Expected frozen v2 split and seed 123.")
 if a.smoke: tr,va,ev=split["train"][:4],split["validation"][:1],split["validation"][1:2]; epochs,per=1,2; label="smoke_validation_1"
 else: tr,va,ev=split["train"],split["validation"],split["validation"]; epochs,per=200,98; label="validation"
 boundary(split,ev); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); fitids=list(dict.fromkeys(tr+va)); loaded={i:base.frame_data(base.load_cold_trajectory(i)) for i in fitids}; norm=fit([loaded[i] for i in tr]); graphs={i:node_graph(loaded[i],protocol["contact_cutoff"]) for i in fitids}; model=InitialMotionBoxResponse(); torch.nn.init.zeros_(model.decoder[-1].weight); torch.nn.init.zeros_(model.decoder[-1].bias); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6); rng=np.random.default_rng(a.seed); best,bstate,stale=float("inf"),None,0; hist=[]; steps=0; beta=10.
 for epoch in range(epochs):
  model.train(); losses=[]; thissteps=0
  for batch in batches(tr,rng):
   opt.zero_grad()
   for i in batch:
    hs=np.arange(3,101) if per==98 else rng.choice(np.arange(3,101),size=per,replace=False); out,_=predict(model,graphs[i],loaded[i],hs,norm,beta); truth=loaded[i][0]["box_tensor"][1:]+targets(loaded[i],hs); loss=((out-truth)/norm["residual_std"]).square().mean(); (loss/len(batch)).backward(); losses.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); thissteps+=1; steps+=1
  model.eval()
  with torch.no_grad(): val=float(np.mean([(predict(model,graphs[i],loaded[i],[20,50,100],norm,beta)[0]-(loaded[i][0]["box_tensor"][1:]+targets(loaded[i],[20,50,100]))).square().mean().item() for i in va]))
  hist.append({"epoch":epoch+1,"training_residual_scaled_yz_box_mse":float(np.mean(losses)),"validation_raw_yz_box_length_mse":val,"optimizer_steps_this_epoch":thissteps,"optimizer_steps_cumulative":steps})
  if val<best: best,bstate,stale=val,{n:v.detach().clone() for n,v in model.state_dict().items()},0
  else: stale+=1
  if epoch+1>=50 and stale>=20: break
 model.load_state_dict(bstate); a.output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":bstate,"normalizers":norm},a.output/"checkpoint.pt")
 with (a.output/"training_history.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
 rows=[]; ols=[]; model.eval()
 with torch.no_grad():
  for i in ev:
   frames=loaded.get(i) or base.frame_data(base.load_cold_trajectory(i)); graph=node_graph(frames,protocol["contact_cutoff"]); e=model.encode(*graph); hs=list(range(3,101)); pred,control=predict(model,graph,frames,hs,norm,beta,e)
   for h,y,o in zip(hs,pred,control):
    for yz,kind in ((y,rows),(o,ols)):
     pos,box=affine(frames,h,yz); py,pz=base.aligned_p_ratios(frames[0]["pos"],pos,box); ty,tz=base.aligned_p_ratios(frames[0]["pos"],frames[h]["pos"],frames[h]["box_tensor"]); kind.append({"network_id":i,"frame":h,"position_mse_physical_minimum_image":base.minimum_image(pos-frames[h]["pos"],frames[h]["box_tensor"]).square().mean().item(),"yz_box_length_mse":(yz-frames[h]["box_tensor"][1:]).square().mean().item(),"pred_p_y_over_x":py,"true_p_y_over_x":ty,"pred_p_z_over_x":pz,"true_p_z_over_x":tz})
 for name,data in (("per_network_per_frame.csv",rows),("ols_per_network_per_frame.csv",ols)):
  with (a.output/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
 summary={"variant":protocol["variant"],"seed":a.seed,"smoke":a.smoke,"evaluation_split":label,"best_validation_raw_yz_box_length_mse":best,"epochs_completed":epoch+1,"optimizer_steps_total":steps,"test":{}}
 for h in protocol["evaluation"]["horizons"]: summary["test"][str(h)]={**metrics([r for r in rows if r["frame"]==h]),"ols_control":metrics([r for r in ols if r["frame"]==h])}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); dataids=list(dict.fromkeys(fitids+ev)); recipe={"runner_sha256":sha(Path(__file__)),"base_v4_runner_sha256":sha(V4),"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha(split_path),"model_source_sha256":sha(ROOT/"src/lss/dynamics/initial_motion_response.py"),"loader_sha256":sha(ROOT/"src/lss/dynamics/cold_graphs.py"),"cohort_sha256":sha(base.COLD_GRAPH_ROOT/"clean_400.json"),"trajectory_sha256":{i:sha(base.COLD_GRAPH_ROOT/f"{i}.pt") for i in dataids},"ols_and_normalizers":{k:v.tolist() for k,v in norm.items()},"seed":a.seed,"smoke":a.smoke,"train_ids":tr,"validation_ids":va,"evaluation_split":label,"evaluation_ids":ev}; (a.output/"recipe.json").write_text(json.dumps(recipe,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
