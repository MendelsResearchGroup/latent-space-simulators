"""Frozen validation-only reference-anchored total-state cold-rubber screen."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, random
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_PATH = ROOT / "hpc/experiments/graphs_cold_global_v2/run.py"
spec = importlib.util.spec_from_file_location("graphs_cold_global_v2_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)
DT = base.DT

def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def targets(initial, following):
    """Only learned targets: reference-anchored total fractional/yz-box state."""
    reference_fractional = base.fractional_coordinates(initial["pos"], initial["box_tensor"])
    following_fractional = base.fractional_coordinates(following["pos"], following["box_tensor"])
    return base.unit_minimum_image(following_fractional - reference_fractional), following["box_tensor"][1:] - initial["box_tensor"][1:]

def fit_normalizers(train, cutoff, seed):
    rng = np.random.default_rng(seed); node_inputs=[]; global_inputs=[]; edge_inputs=[]; node_targets=[]; box_targets=[]
    for frames in train:
        for center in rng.choice(np.arange(2, 399), size=16, replace=False):
            previous, current, following = frames[center - 1], frames[center], frames[center + 1]
            node, box = targets(frames[0], following); edge, _ = base.raw_edge(current, cutoff)
            node_inputs.append(base.raw_node(previous, current, frames[0])); global_inputs.append(base.raw_global(previous, current, frames[0])); edge_inputs.append(edge); node_targets.append(node); box_targets.append(box)
    def stats(values, stack=False):
        joined=torch.stack(values) if stack else torch.cat(values)
        return joined.mean(0), joined.std(0).clamp_min(1e-12)
    nmean,nstd=stats(node_inputs); gmean,gstd=stats(global_inputs, True); emean,estd=stats(edge_inputs); tmean,tstd=stats(node_targets); bmean,bstd=stats(box_targets, True)
    return {"node_input_mean":nmean,"node_input_std":nstd,"global_input_mean":gmean,"global_input_std":gstd,"edge_input_mean":emean,"edge_input_std":estd,"node_target_mean":tmean,"node_target_std":tstd,"box_target_mean":bmean,"box_target_std":bstd}

def advance(model, previous, current, initial, normalizers, cutoff, hidden_state=None):
    node, yz_box, next_hidden = model(*base.model_inputs(previous, current, initial, normalizers, cutoff), hidden_state=hidden_state)
    total_fractional = node * normalizers["node_target_std"] + normalizers["node_target_mean"]
    total_yz_box = yz_box * normalizers["box_target_std"] + normalizers["box_target_mean"]
    reference_fractional = base.fractional_coordinates(initial["pos"], initial["box_tensor"])
    next_fractional = base.unit_minimum_image(reference_fractional + total_fractional)
    x_box = current["box_tensor"][0] + (current["box_tensor"][0] - previous["box_tensor"][0])
    box = torch.cat((x_box.reshape(1), initial["box_tensor"][1:] + total_yz_box))
    if torch.any(box <= 0): raise ValueError("Predicted non-positive box length.")
    return {**current, "pos": next_fractional * box, "box_tensor": box, "time": current["time"] + DT}, next_hidden, node, yz_box

def standardized_total_state_loss(node_prediction, box_prediction, initial, following, normalizers):
    node_true, box_true = targets(initial, following)
    return (node_prediction - base.standardize(node_true, normalizers["node_target_mean"], normalizers["node_target_std"])).square().mean() + (box_prediction - base.standardize(box_true, normalizers["box_target_mean"], normalizers["box_target_std"])).square().mean()

def rollout(model, frames, horizon, normalizers, cutoff):
    previous,current,hidden=frames[1],frames[2],None
    for _ in range(3, horizon + 1):
        predicted,hidden,_,_=advance(model, previous, current, frames[0], normalizers, cutoff, hidden)
        previous,current=current,predicted
    return current

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--variant", choices=("anchored_contact","anchored_contact_recurrent"), required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    protocol=json.loads((HERE/"protocol.json").read_text()); split_path=(HERE/protocol["split_manifest"]).resolve(); split=json.loads(split_path.read_text())
    all_ids=[identifier for key in ("train","validation","final_test") for identifier in split[key]]
    if len(split["train"])!=48 or len(split["validation"])!=12 or len(split["final_test"])!=12 or len(set(all_ids))!=72: raise ValueError("Expected frozen disjoint v2 48/12/12 split.")
    if args.seed != 123: raise ValueError("Frozen screen permits seed 123 only.")
    if args.smoke:
        train_ids,validation_ids,evaluation_ids=split["train"][:1],split["validation"][:1],split["validation"][1:2]; phase1_epochs=phase2_epochs=1; centers,windows,unroll=2,1,2; evaluation_label="smoke_validation_1"
    else:
        train_ids,validation_ids,evaluation_ids=split["train"],split["validation"],split["validation"]; phase1_epochs=protocol["phase1"]["epochs"]; phase2_epochs=protocol["phase2"]["max_epochs"]; centers=protocol["phase1"]["centers_per_train_trajectory_epoch"]; windows=protocol["phase2"]["windows_per_train_trajectory_epoch"]; unroll=protocol["phase2"]["unroll_steps"]; evaluation_label="validation"
    if set(evaluation_ids)-set(split["validation"]): raise RuntimeError("Validation-only harness must never load final_test.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    fit_ids=list(dict.fromkeys(train_ids+validation_ids)); loaded={identifier:base.frame_data(base.load_cold_trajectory(identifier)) for identifier in fit_ids}
    variant=protocol["variants"][args.variant]; cutoff=variant["contact_cutoff"]; normalizers=fit_normalizers([loaded[x] for x in train_ids],cutoff,protocol["normalization"]["seed"])
    model=base.FractionalContactSimulator(**protocol["architecture"], recurrent=variant["recurrent"]); rng=np.random.default_rng(args.seed); validation_rng=np.random.default_rng(protocol["phase1"]["validation_seed"])
    validation_centers={identifier:validation_rng.choice(np.arange(2,399),size=(2 if args.smoke else protocol["phase1"]["validation_centers_per_trajectory"]),replace=False) for identifier in validation_ids}; history=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=protocol["phase1"]["learning_rate"],weight_decay=protocol["phase1"]["weight_decay"]); best1,best1_state=float("inf"),None
    for epoch in range(phase1_epochs):
        model.train(); optimizer.zero_grad(); losses=[]; samples=base.sample_centers(train_ids,centers,rng)
        for number,(identifier,center) in enumerate(samples,1):
            frames=loaded[identifier]; node,box,_=model(*base.model_inputs(frames[center-1],frames[center],frames[0],normalizers,cutoff))
            loss=standardized_total_state_loss(node,box,frames[0],frames[center+1],normalizers); (loss/protocol["phase1"]["gradient_accumulation"]).backward(); losses.append(float(loss.detach()))
            if number%protocol["phase1"]["gradient_accumulation"]==0 or number==len(samples): torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["phase1"]["gradient_clip_norm"]); optimizer.step(); optimizer.zero_grad()
        model.eval()
        with torch.no_grad():
            validation=float(np.mean([base.physical_state_mse(advance(model,loaded[i][t-1],loaded[i][t],loaded[i][0],normalizers,cutoff)[0],loaded[i][t+1]).item() for i in validation_ids for t in validation_centers[i]]))
        history.append({"phase":"one_step","epoch":epoch+1,"training_standardized_total_state_mse":float(np.mean(losses)),"validation_physical_state_mse":validation})
        if validation<best1: best1,best1_state=validation,{name:value.detach().clone() for name,value in model.state_dict().items()}
    if best1_state is None: raise RuntimeError("No phase-1 checkpoint.")
    model.load_state_dict(best1_state); optimizer=torch.optim.Adam(model.parameters(),lr=protocol["phase2"]["learning_rate"],weight_decay=protocol["phase2"]["weight_decay"]); best2,best2_state,stale=float("inf"),None,0; selection_horizons=(4,) if args.smoke else tuple(protocol["phase2"]["selection_horizons"])
    for epoch in range(phase2_epochs):
        model.train(); losses=[]
        for identifier,start in base.sample_windows(train_ids,windows,unroll,rng):
            frames=loaded[identifier]; previous,current,hidden=frames[start-1],frames[start],None; loss=torch.zeros(())
            for offset in range(1,unroll+1):
                predicted,hidden,node,box=advance(model,previous,current,frames[0],normalizers,cutoff,hidden)
                loss=loss+standardized_total_state_loss(node,box,frames[0],frames[start+offset],normalizers); previous,current=current,predicted
            loss=loss/unroll; optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["phase2"]["gradient_clip_norm"]); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): validation=float(np.mean([base.physical_state_mse(rollout(model,loaded[i],h,normalizers,cutoff),loaded[i][h]).item() for i in validation_ids for h in selection_horizons]))
        history.append({"phase":"autonomous","epoch":epoch+1,"training_standardized_total_state_mse":float(np.mean(losses)),"validation_autonomous_physical_state_mse":validation,"selection_horizons":list(selection_horizons)})
        if validation<best2: best2,best2_state,stale=validation,{name:value.detach().clone() for name,value in model.state_dict().items()},0
        else: stale+=1
        if stale>=protocol["phase2"]["patience"]: break
    if best2_state is None: raise RuntimeError("No phase-2 checkpoint.")
    model.load_state_dict(best2_state); args.output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best2_state,"normalizers":normalizers},args.output/"checkpoint.pt")
    with (args.output/"training_history.csv").open("w",newline="") as handle:
        fields=list(dict.fromkeys(key for row in history for key in row)); writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(history)
    rows=[]; model.eval()
    with torch.no_grad():
        for identifier in evaluation_ids:
            frames=loaded.get(identifier) or base.frame_data(base.load_cold_trajectory(identifier)); previous,current,hidden=frames[1],frames[2],None
            for frame in range(3,400):
                predicted,hidden,_,_=advance(model,previous,current,frames[0],normalizers,cutoff,hidden); pred_y,pred_z=base.aligned_p_ratios(frames[0]["pos"],predicted["pos"],predicted["box_tensor"]); true_y,true_z=base.aligned_p_ratios(frames[0]["pos"],frames[frame]["pos"],frames[frame]["box_tensor"])
                rows.append({"network_id":identifier,"frame":frame,"position_mse_physical_minimum_image":base.position_mse(predicted,frames[frame]).item(),"pred_p_y_over_x":pred_y,"true_p_y_over_x":true_y,"pred_p_z_over_x":pred_z,"true_p_z_over_x":true_z}); previous,current=current,predicted
    with (args.output/"per_network_per_frame.csv").open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={"variant":args.variant,"seed":args.seed,"smoke":args.smoke,"evaluation_split":evaluation_label,"best_phase1_validation_one_step_physical_state_mse":best1,"best_phase2_validation_autonomous_physical_state_mse":best2,"phase2_epochs_completed":epoch+1,"test":{}}
    for horizon in protocol["evaluation"]["horizons"]:
        selected=[row for row in rows if row["frame"]==horizon]; yr2,yvalid=base.r2(selected,"true_p_y_over_x","pred_p_y_over_x"); zr2,zvalid=base.r2(selected,"true_p_z_over_x","pred_p_z_over_x")
        summary["test"][str(horizon)]={"position_mse_physical_minimum_image":float(np.mean([row["position_mse_physical_minimum_image"] for row in selected])),"p_ratio_r2_y_over_x":yr2,"p_ratio_r2_z_over_x":zr2,"p_ratio_y_valid_total":[yvalid,len(selected)],"p_ratio_z_valid_total":[zvalid,len(selected)]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    data_ids=list(dict.fromkeys(fit_ids+evaluation_ids)); recipe={"runner_sha256":sha256(Path(__file__)),"base_v2_runner_sha256":sha256(BASE_PATH),"protocol_sha256":sha256(HERE/"protocol.json"),"split_manifest_sha256":sha256(split_path),"loader_sha256":sha256(ROOT/"src/lss/dynamics/cold_graphs.py"),"model_source_sha256":sha256(ROOT/"src/lss/dynamics/fractional_contact_simulator.py"),"cohort_sha256":sha256(base.COLD_GRAPH_ROOT/"clean_400.json"),"trajectory_sha256":{i:sha256(base.COLD_GRAPH_ROOT/f"{i}.pt") for i in data_ids},"variant":args.variant,"contact_cutoff":cutoff,"recurrent":variant["recurrent"],"seed":args.seed,"smoke":args.smoke,"normalizer":{"train_ids":train_ids,**protocol["normalization"]},"train_ids":train_ids,"validation_ids":validation_ids,"evaluation_split":evaluation_label,"evaluation_ids":evaluation_ids}
    (args.output/"recipe.json").write_text(json.dumps(recipe,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
