"""Fixed local-versus-global cold-rubber acceleration experiment."""
from __future__ import annotations

import argparse, csv, json, random, hashlib, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/lss/dynamics"))

from cold_graphs import load_cold_trajectory, position_p_ratios, rebuild_edge_features
from global_context_simulator import LocalGlobalLocalSimulator

HERE = Path(__file__).resolve().parent
DT = 200.0  # centers 2..398 all have this serialized cadence
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def mi(delta, box): return delta - torch.round(delta / box) * box
def frame_data(frames):
    return [{k: getattr(f, k) for k in ("pos", "edge_index", "edge_attr", "box_tensor", "atom_types", "time")} for f in frames]
def inputs(initial_box, ref, previous, current, norm):
    box0 = initial_box
    velocity = mi(current["pos"] - previous["pos"], previous["box_tensor"]) / (current["time"] - previous["time"])
    raw=torch.cat((ref["pos"] / box0, current["pos"] / box0, velocity / box0), 1)
    return (raw-norm["node_input_mean"])/norm["node_input_std"]
def target(previous, current, following):
    return ((mi(following["pos"]-current["pos"],current["box_tensor"])-mi(current["pos"]-previous["pos"],previous["box_tensor"]))/DT**2,
            (following["box_tensor"] - 2 * current["box_tensor"] + previous["box_tensor"]) / DT**2)
def advance(model, initial_box, ref, previous, current, norm):
    edge = rebuild_edge_features(current["pos"], current["edge_index"], current["box_tensor"], current["edge_attr"][:, 4])
    edge=(edge-norm["edge_input_mean"])/norm["edge_input_std"]
    n, b = model(inputs(initial_box, ref, previous, current, norm), current["atom_types"] - 1, edge, current["edge_index"])
    n = n * norm["node_std"] + norm["node_mean"]; b = b * norm["box_std"] + norm["box_mean"]
    pos = current["pos"] + mi(current["pos"] - previous["pos"], current["box_tensor"]) + n * DT**2
    box = current["box_tensor"] + (current["box_tensor"] - previous["box_tensor"]) + b * DT**2
    if torch.any(box <= 0): raise ValueError("Predicted non-positive box length.")
    pos=mi(pos,box)
    return {**current, "pos": pos, "box_tensor": box, "time": current["time"] + DT}
def mse_state(pred, truth):
    return (mi(pred["pos"] - truth["pos"], truth["box_tensor"]).square().mean() + (pred["box_tensor"] - truth["box_tensor"]).square().mean()).item()
def mse_position(pred, truth):
    return mi(pred["pos"] - truth["pos"], truth["box_tensor"]).square().mean().item()
def norms(trajectories):
    ns=[]; bs=[]; xs=[]; es=[]
    for fs in trajectories:
        for t in range(2,399):
            n,b=target(fs[t-1],fs[t],fs[t+1]); ns.append(n); bs.append(b)
            xs.append(torch.cat((fs[0]["pos"]/fs[0]["box_tensor"],fs[t]["pos"]/fs[0]["box_tensor"],mi(fs[t]["pos"]-fs[t-1]["pos"],fs[t-1]["box_tensor"])/(fs[t]["time"]-fs[t-1]["time"])/fs[0]["box_tensor"]),1))
            es.append(rebuild_edge_features(fs[t]["pos"],fs[t]["edge_index"],fs[t]["box_tensor"],fs[t]["edge_attr"][:,4]))
    n=torch.cat(ns); b=torch.stack(bs); x=torch.cat(xs); e=torch.cat(es)
    return {"node_mean":n.mean(0),"node_std":n.std(0).clamp_min(1e-12),"box_mean":b.mean(0),"box_std":b.std(0).clamp_min(1e-12),"node_input_mean":x.mean(0),"node_input_std":x.std(0).clamp_min(1e-12),"edge_input_mean":e.mean(0),"edge_input_std":e.std(0).clamp_min(1e-12)}
def choose(ids, count, rng):
    samples=[(name, int(t)) for name in ids for t in rng.choice(np.arange(2,399), size=count, replace=False)]
    rng.shuffle(samples)
    return samples
def r2(pairs):
    valid=[(true,pred) for true,pred in pairs if np.isfinite(true) and np.isfinite(pred)]
    if len(valid) < 2: return float("nan"),len(valid)
    true,pred=np.asarray(valid).T; sst=np.square(true-true.mean()).sum()
    if sst <= 0: return float("nan"),len(valid)
    return float(1-np.square(pred-true).sum()/sst),len(valid)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=("local","global"),required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--smoke",action="store_true"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    protocol=json.loads((HERE/"protocol.json").read_text()); split=json.loads((HERE/"split_manifest.json").read_text())
    if len(set(sum((split[x] for x in ("train","validation","final_test")),[]))) != 24: raise ValueError("Split is not disjoint.")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    evaluation_ids=split["final_test"]; evaluation_label="final_test"
    if a.smoke:
        # Smoke integration evaluation is held out from smoke training, but never touches final test.
        split={"train":split["train"][:1],"validation":split["validation"][:1]}
        evaluation_ids=json.loads((HERE/"split_manifest.json").read_text())["validation"][1:2]
        evaluation_label="smoke_eval"
    # Final-test files are not loaded until after state-only checkpoint selection is frozen.
    fit_ids=list(dict.fromkeys(split["train"]+split["validation"]))
    loaded={name:frame_data(load_cold_trajectory(name)) for name in fit_ids}
    train=[loaded[x] for x in split["train"]]; norm=norms(train)
    attention=protocol["models"][a.model]["attention_layers"]
    model=LocalGlobalLocalSimulator(node_feature_dim=9,edge_feature_dim=5,atom_type_count=2,hidden_dim=64,attention_heads=4,message_steps_each_side=2,attention_layers=attention)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-6); rng=np.random.default_rng(a.seed); best=float("inf"); stale=0; best_state=None
    epochs=1 if a.smoke else int(protocol["training"]["max_epochs"])
    centers=4 if a.smoke else int(protocol["training"]["centers_per_train_trajectory_epoch"])
    accumulation=int(protocol["training"]["gradient_accumulation"])
    history=[]
    for epoch in range(epochs):
        model.train(); opt.zero_grad(); sampled=choose(split["train"],centers,rng); training_loss=0.0
        for sample_index,(name,t) in enumerate(sampled, start=1):
            fs=loaded[name]; ntrue,btrue=target(fs[t-1],fs[t],fs[t+1]); edge=rebuild_edge_features(fs[t]["pos"],fs[t]["edge_index"],fs[t]["box_tensor"],fs[t]["edge_attr"][:,4]); edge=(edge-norm["edge_input_mean"])/norm["edge_input_std"]; npred,bpred=model(inputs(fs[0]["box_tensor"],fs[0],fs[t-1],fs[t],norm),fs[t]["atom_types"]-1,edge,fs[t]["edge_index"]); loss=(npred-(ntrue-norm["node_mean"])/norm["node_std"]).square().mean()+(bpred-(btrue-norm["box_mean"])/norm["box_std"]).square().mean(); training_loss += loss.item()
            # Each update averages four independent graph states; no graph batch is materialized.
            (loss/accumulation).backward()
            if sample_index % accumulation == 0 or sample_index == len(sampled):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad()
        model.eval(); vals=[]
        with torch.no_grad():
            for name in split["validation"]:
                fs=loaded[name]
                for t in range(2,399): vals.append(mse_state(advance(model,fs[0]["box_tensor"],fs[0],fs[t-1],fs[t],norm),fs[t+1]))
        value=float(np.mean(vals))
        history.append({"epoch":epoch+1,"training_standardized_acceleration_mse":training_loss/len(sampled),"validation_one_step_combined_state_mse":value})
        if value < best: best=value; stale=0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale += 1
        if stale >= 5: break
    if best_state is None: raise RuntimeError("No validation checkpoint.")
    model.load_state_dict(best_state); a.output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best_state,"normalizers":norm},a.output/"checkpoint.pt")
    with (a.output/"training_history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    rows=[]; horizons=set(protocol["test"]["horizons"])
    model.eval()
    with torch.no_grad():
        for name in evaluation_ids:
            fs=frame_data(load_cold_trajectory(name)); previous,current=fs[1],fs[2]
            for h in range(3,400):
                predicted=advance(model,fs[0]["box_tensor"],fs[0],previous,current,norm); previous,current=current,predicted
                py,pz=position_p_ratios(fs[0]["pos"].numpy(),predicted["pos"].numpy())
                ty,tz=position_p_ratios(fs[0]["pos"].numpy(),fs[h]["pos"].numpy())
                rows.append({"network_id":name,"frame":h,"position_mse_physical_minimum_image":mse_position(predicted,fs[h]),"pred_p_y_over_x":py,"true_p_y_over_x":ty,"pred_p_z_over_x":pz,"true_p_z_over_x":tz})
    with (a.output/"per_network_per_frame.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"model":a.model,"seed":a.seed,"smoke":a.smoke,"evaluation_split":evaluation_label,"best_validation_one_step_combined_state_mse":best,"epochs_completed":epoch+1,"test":{}}
    for h in protocol["test"]["horizons"]:
        at=[x for x in rows if x["frame"]==h]
        yr2,yvalid=r2([(x["true_p_y_over_x"],x["pred_p_y_over_x"]) for x in at]); zr2,zvalid=r2([(x["true_p_z_over_x"],x["pred_p_z_over_x"]) for x in at])
        summary["test"][str(h)]={"position_mse_physical_minimum_image":float(np.mean([x["position_mse_physical_minimum_image"] for x in at])),"p_ratio_r2_y_over_x":yr2 if np.isfinite(yr2) else None,"p_ratio_r2_z_over_x":zr2 if np.isfinite(zr2) else None,"p_ratio_y_valid_total":[yvalid,len(at)],"p_ratio_z_valid_total":[zvalid,len(at)]}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    source_root=Path("/rg/mendels_prj/s.sergey/graphs_cold")
    data_ids=list(dict.fromkeys(fit_ids+evaluation_ids))
    (a.output/"recipe.json").write_text(json.dumps({"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha(HERE/"split_manifest.json"),"runner_sha256":sha(HERE/"run.py"),"loader_sha256":sha(ROOT/"src/lss/dynamics/cold_graphs.py"),"model_source_sha256":sha(ROOT/"src/lss/dynamics/global_context_simulator.py"),"cohort_sha256":sha(source_root/"clean_400.json"),"registry_sha256":sha(source_root/"registry.csv"),"trajectory_sha256":{name:sha(source_root/f"{name}.pt") for name in data_ids},"model":a.model,"seed":a.seed,"smoke":a.smoke,"train_ids":split["train"],"validation_ids":split["validation"],"evaluation_split":evaluation_label,"evaluation_ids":evaluation_ids},indent=2,sort_keys=True)+"\n")
if __name__ == "__main__": main()
