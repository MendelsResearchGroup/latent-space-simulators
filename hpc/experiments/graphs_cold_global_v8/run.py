from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,random,os
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; V4=ROOT/"hpc/experiments/graphs_cold_global_v4/run.py"; spec=importlib.util.spec_from_file_location("v4",V4); v4=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(v4); base=v4.base
from node_only_global_simulator import NodeOnlyGlobalSimulator
DT=200.
CONTACT_CUTOFF=1.5
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mi(x,b):return x-torch.round(x/b)*b
def boundary(s,x):
 if set(x)-set(s["validation"]):raise RuntimeError("Validation-only harness must never load final_test.")
def align(fs):
 ref=fs[0]["pos"]; return [{**f,"pos":ref+mi(f["pos"]-ref,f["box_tensor"])} for f in fs]
def box(fs,t):
 d=(fs[1]["box_tensor"][0]-fs[0]["box_tensor"][0])/(fs[1]["time"]-fs[0]["time"]); return torch.cat(((fs[0]["box_tensor"][0]+t*DT*d).reshape(1),fs[0]["box_tensor"][1:]))
def edge(f,b):return base.build_interaction_graph(f["pos"],f["edge_index"],b,f["edge_attr"][:,4],contact_cutoff=CONTACT_CUTOFF)
def raw(fs,p,c):
 b0=fs[0]["box_tensor"]; return torch.cat((fs[0]["pos"]/b0,(c["pos"]-fs[0]["pos"])/b0,(c["pos"]-p["pos"])/b0),1)
def target(p,c,n):return (n["pos"]-2*c["pos"]+p["pos"])/DT**2
def fit(train):
 xs=[];es=[];ys=[]
 rng=np.random.default_rng(20260918)
 for fs in train:
  for t in rng.choice(np.arange(1,100),16,replace=False):
   e,_=edge(fs[t],box(fs,t));xs.append(raw(fs,fs[t-1],fs[t]));es.append(e);ys.append(target(fs[t-1],fs[t],fs[t+1]))
 def st(v):x=torch.cat(v);return x.mean(0),x.std(0).clamp_min(1e-12)
 xm,xs_=st(xs);em,es_=st(es);_,ys_=st(ys);return {"xm":xm,"xs":xs_,"em":em,"es":es_,"ys":ys_}
def inp(fs,p,c,t,n):
 e,i=edge(c,box(fs,t));return (raw(fs,p,c)-n["xm"])/n["xs"],c["atom_types"]-1,(e-n["em"])/n["es"],i
def integrate_constant_velocity(previous,current,normalized_acceleration,acceleration_std):
 return 2*current-previous+normalized_acceleration*acceleration_std*DT**2
def step(m,fs,p,c,t,n):return {**c,"pos":integrate_constant_velocity(p["pos"],c["pos"],m(*inp(fs,p,c,t,n)),n["ys"]),"time":c["time"]+DT}
def roll(m,fs,h,n):
 p,c=fs[0],fs[1]
 for t in range(2,h+1):p,c=c,step(m,fs,p,c,t-1,n)
 return c
def main():
 a=argparse.ArgumentParser();a.add_argument("--seed",type=int,required=True);a.add_argument("--smoke",action="store_true");a.add_argument("--output",type=Path,required=True);z=a.parse_args();P=json.loads((HERE/"protocol.json").read_text());global CONTACT_CUTOFF;CONTACT_CUTOFF=P["contact_cutoff"];sp=(HERE/P["split_manifest"]).resolve();s=json.loads(sp.read_text());
 if z.seed!=123 or [len(s[k]) for k in ("train","validation","final_test")] != [48,12,12]:raise ValueError("frozen split/seed")
 tr,va,ev=(s["train"][:1],s["validation"][:1],s["validation"][1:2]) if z.smoke else (s["train"],s["validation"],s["validation"]);boundary(s,ev);epochs=1 if z.smoke else 10; centers=2 if z.smoke else 24; unroll=2 if z.smoke else 20; windows=1 if z.smoke else 3;random.seed(123);np.random.seed(123);torch.manual_seed(123);loaded={i:align(base.frame_data(base.load_cold_trajectory(i))) for i in dict.fromkeys(tr+va)};n=fit([loaded[i] for i in tr]);m=NodeOnlyGlobalSimulator();rng=np.random.default_rng(123);hist=[];opt=torch.optim.Adam(m.parameters(),lr=3e-4,weight_decay=1e-6);best=float("inf");state=None
 vrng=np.random.default_rng(P["phase1"]["validation_seed"]);vc={i:vrng.choice(np.arange(1,100),2 if z.smoke else P["phase1"]["validation_centers"],replace=False) for i in va}
 for ep in range(epochs):
  m.train();losses=[];samples=[(i,int(t)) for i in tr for t in rng.choice(np.arange(1,100),centers,replace=False)];rng.shuffle(samples);opt.zero_grad()
  for q,(i,t) in enumerate(samples,1):
   fs=loaded[i];loss=(m(*inp(fs,fs[t-1],fs[t],t,n))-target(fs[t-1],fs[t],fs[t+1])/n["ys"]).square().mean();(loss/4).backward();losses.append(float(loss));
   if q%4==0 or q==len(samples):torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad()
  m.eval();v=float(np.mean([mi(step(m,loaded[i],loaded[i][t-1],loaded[i][t],t,n)["pos"]-loaded[i][t+1]["pos"],box(loaded[i],t+1)).square().mean().item() for i in va for t in vc[i]]));hist.append({"phase":"one_step","epoch":ep+1,"train_acceleration_mse":np.mean(losses),"validation_position_mse":v});
  if v<best:best,state=v,{k:x.detach().clone() for k,x in m.state_dict().items()}
 m.load_state_dict(state);opt=torch.optim.Adam(m.parameters(),lr=1e-4,weight_decay=1e-6);best2=float("inf");stale=0
 for ep in range(1 if z.smoke else 10):
  m.train();ls=[]
  samples=[(i,int(start)) for i in tr for start in rng.choice(np.arange(1,100-unroll),windows,replace=False)];rng.shuffle(samples)
  for i,start in samples:
   fs=loaded[i];p,c=fs[start-1],fs[start];loss=0
   for o in range(1,unroll+1):pr=step(m,fs,p,c,start+o-1,n);loss=loss+(mi(pr["pos"]-fs[start+o]["pos"],box(fs,start+o))/(n["ys"]*DT**2)).square().mean();p,c=c,pr
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();ls.append(float(loss))
  m.eval();val=float(np.mean([mi(roll(m,loaded[i],h,n)["pos"]-loaded[i][h]["pos"],box(loaded[i],h)).square().mean().item() for i in va for h in ((20,50,100) if not z.smoke else (2,))]));hist.append({"phase":"autonomous","epoch":ep+1,"train_position_loss":np.mean(ls),"validation_position_mse":val});
  if val<best2:best2,state2=val,{k:x.detach().clone() for k,x in m.state_dict().items()};stale=0
  else:stale+=1
  if stale>=4:break
 m.load_state_dict(state2);z.output.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":state2,"normalizers":n},z.output/"checkpoint.pt");
 with (z.output/"training_history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list({k:1 for x in hist for k in x}));w.writeheader();w.writerows(hist)
 rows=[]
 with torch.no_grad():
  for i in ev:
   fs=align(base.frame_data(base.load_cold_trajectory(i)));previous,current=fs[0],fs[1]
   for h in range(2,101):
    predicted=step(m,fs,previous,current,h-1,n);previous,current=current,predicted;py,_=base.aligned_p_ratios(fs[0]["pos"],predicted["pos"],box(fs,h));ty,_=base.aligned_p_ratios(fs[0]["pos"],fs[h]["pos"],fs[h]["box_tensor"]);rows.append({"network_id":i,"frame":h,"position_mse_physical_minimum_image":mi(predicted["pos"]-fs[h]["pos"],fs[h]["box_tensor"]).square().mean().item(),"pred_p_y_over_x":py,"true_p_y_over_x":ty})
 with (z.output/"per_network_per_frame.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={"seed":123,"smoke":z.smoke,"evaluation_split":"smoke_validation_1" if z.smoke else "validation","best_phase1":best,"best_phase2":best2,"test":{}}
 for h in P["evaluation"]["horizons"]:
  r=[x for x in rows if x["frame"]==h];rr,vv=base.r2(r,"true_p_y_over_x","pred_p_y_over_x");summary["test"][str(h)]={"position_mse_physical_minimum_image":float(np.mean([x["position_mse_physical_minimum_image"] for x in r])),"p_ratio_r2_y_over_x":rr,"p_ratio_y_valid_total":[vv,len(r)]}
 (z.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");dataids=list(dict.fromkeys(tr+va+ev));recipe={"runner_sha256":sha(Path(__file__)),"base_v4_runner_sha256":sha(V4),"base_v2_runner_sha256":sha(v4.BASE_PATH),"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha(sp),"model_source_sha256":sha(ROOT/"src/lss/dynamics/node_only_global_simulator.py"),"global_context_source_sha256":sha(ROOT/"src/lss/dynamics/global_context_simulator.py"),"loader_sha256":sha(ROOT/"src/lss/dynamics/cold_graphs.py"),"cohort_sha256":sha(base.COLD_GRAPH_ROOT/"clean_400.json"),"trajectory_sha256":{i:sha(base.COLD_GRAPH_ROOT/f"{i}.pt") for i in dataids},"pbs_job_id":os.getenv("PBS_JOBID"),"seed":123,"train_ids":tr,"selection_validation_ids":va,"evaluation_ids":ev,"evaluation_split":summary["evaluation_split"],"normalizers":{k:v.tolist() for k,v in n.items()}};(z.output/"recipe.json").write_text(json.dumps(recipe,indent=2)+"\n")
if __name__=="__main__":main()
