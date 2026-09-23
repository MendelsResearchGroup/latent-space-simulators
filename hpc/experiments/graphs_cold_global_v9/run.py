from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,random
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];V8=ROOT/"hpc/experiments/graphs_cold_global_v8/run.py";s=importlib.util.spec_from_file_location("v8",V8);v8=importlib.util.module_from_spec(s);assert s.loader;s.loader.exec_module(v8);base=v8.base
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,required=True);p.add_argument("--smoke",action="store_true");p.add_argument("--output",type=Path,required=True);a=p.parse_args();P=json.loads((HERE/"protocol.json").read_text());sp=(HERE/P["split_manifest"]).resolve();split=json.loads(sp.read_text())
 if a.seed!=123 or [len(split[x]) for x in ("train","validation","final_test")]!=[48,12,12]:raise ValueError("frozen split/seed")
 tr,va,ev=(split["train"][:1],split["validation"][:1],split["validation"][1:2]) if a.smoke else (split["train"],split["validation"],split["validation"]);v8.boundary(split,ev);epochs=1 if a.smoke else 10;centers=2 if a.smoke else 24;random.seed(123);np.random.seed(123);torch.manual_seed(123);loaded={i:v8.align(base.frame_data(base.load_cold_trajectory(i))) for i in dict.fromkeys(tr+va)};n=v8.fit([loaded[i] for i in tr]);m=v8.NodeOnlyGlobalSimulator();rng=np.random.default_rng(123);vrng=np.random.default_rng(P["phase1"]["validation_seed"]);vc={i:vrng.choice(np.arange(1,100),2 if a.smoke else 32,replace=False) for i in va};opt=torch.optim.Adam(m.parameters(),lr=3e-4,weight_decay=1e-6);best=float("inf");state=None;history=[]
 for ep in range(epochs):
  m.train();losses=[];samples=[(i,int(t)) for i in tr for t in rng.choice(np.arange(1,100),centers,replace=False)];rng.shuffle(samples);opt.zero_grad()
  for q,(i,t) in enumerate(samples,1):
   fs=loaded[i];loss=(m(*v8.inp(fs,fs[t-1],fs[t],t,n))-v8.target(fs[t-1],fs[t],fs[t+1])/n["ys"]).square().mean();(loss/4).backward();losses.append(float(loss.detach()))
   if q%4==0 or q==len(samples):torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();opt.zero_grad()
  m.eval();val=float(np.mean([v8.mi(v8.step(m,loaded[i],loaded[i][t-1],loaded[i][t],t,n)["pos"]-loaded[i][t+1]["pos"],v8.box(loaded[i],t+1)).square().mean().item() for i in va for t in vc[i]]));history.append({"epoch":ep+1,"training_acceleration_mse":float(np.mean(losses)),"validation_one_step_position_mse":val})
  if val<best:best,state=val,{k:x.detach().clone() for k,x in m.state_dict().items()}
 m.load_state_dict(state);a.output.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":state,"normalizers":n},a.output/"checkpoint.pt")
 with (a.output/"training_history.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
 rows=[]
 with torch.no_grad():
  for i in ev:
   fs=v8.align(base.frame_data(base.load_cold_trajectory(i)));previous,current=fs[0],fs[1]
   for h in range(2,101):
    predicted=v8.step(m,fs,previous,current,h-1,n);previous,current=current,predicted;py,_=base.aligned_p_ratios(fs[0]["pos"],predicted["pos"],v8.box(fs,h));ty,_=base.aligned_p_ratios(fs[0]["pos"],fs[h]["pos"],fs[h]["box_tensor"]);rows.append({"network_id":i,"frame":h,"position_mse_physical_minimum_image":v8.mi(predicted["pos"]-fs[h]["pos"],fs[h]["box_tensor"]).square().mean().item(),"pred_p_y_over_x":py,"true_p_y_over_x":ty})
 with (a.output/"per_network_per_frame.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={"seed":123,"smoke":a.smoke,"evaluation_split":"smoke_validation_1" if a.smoke else "validation","best_validation_one_step_position_mse":best,"epochs_completed":epochs,"test":{}}
 for h in P["evaluation"]["horizons"]:
  r=[x for x in rows if x["frame"]==h];rr,v=base.r2(r,"true_p_y_over_x","pred_p_y_over_x");summary["test"][str(h)]={"position_mse_physical_minimum_image":float(np.mean([x["position_mse_physical_minimum_image"] for x in r])),"p_ratio_r2_y_over_x":rr,"p_ratio_y_valid_total":[v,len(r)]}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");ids=list(dict.fromkeys(tr+va+ev));recipe={"runner_sha256":sha(Path(__file__)),"base_v8_runner_sha256":sha(V8),"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha(sp),"model_source_sha256":sha(ROOT/"src/lss/dynamics/node_only_global_simulator.py"),"loader_sha256":sha(ROOT/"src/lss/dynamics/cold_graphs.py"),"cohort_sha256":sha(base.COLD_GRAPH_ROOT/"clean_400.json"),"trajectory_sha256":{i:sha(base.COLD_GRAPH_ROOT/f"{i}.pt") for i in ids},"pbs_job_id":os.getenv("PBS_JOBID"),"train_ids":tr,"selection_validation_ids":va,"evaluation_ids":ev,"evaluation_split":summary["evaluation_split"],"normalizers":{k:v.tolist() for k,v in n.items()}};(a.output/"recipe.json").write_text(json.dumps(recipe,indent=2)+"\n")
if __name__=="__main__":main()
