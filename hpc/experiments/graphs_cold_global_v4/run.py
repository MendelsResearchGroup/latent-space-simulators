"""Frozen validation-only static-graph prescribed-x box-response screen."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, random
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; BASE_PATH=ROOT/"hpc/experiments/graphs_cold_global_v2/run.py"
spec=importlib.util.spec_from_file_location("cold_v2_base",BASE_PATH); base=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(base)
from static_response_head import StaticGlobalBoxResponse
DT=200.0
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def reference_graph(initial, cutoff):
    edges,index=base.raw_edge(initial,cutoff)
    return base.fractional_coordinates(initial["pos"],initial["box_tensor"]),initial["atom_types"]-1,edges,index

def prescribed_x(initial, observed_one, observed_two, horizon):
    delta=(observed_two["box_tensor"][0]-observed_one["box_tensor"][0])/(observed_two["time"]-observed_one["time"])
    return observed_two["box_tensor"][0]+(horizon-2)*DT*delta,delta

def context(initial, observed_one, observed_two, horizon):
    target_x,delta=prescribed_x(initial,observed_one,observed_two,horizon)
    return torch.cat((initial["box_tensor"],target_x.reshape(1),delta.reshape(1)))

def target(initial, frame): return frame["box_tensor"][1:]-initial["box_tensor"][1:]

def fit_normalizers(train, seed, centers_per_trajectory):
    rng=np.random.default_rng(seed); contexts=[]; targets=[]
    for frames in train:
        horizons=np.arange(3,400) if centers_per_trajectory==397 else rng.choice(np.arange(3,400),size=centers_per_trajectory,replace=False)
        for horizon in horizons:
            contexts.append(context(frames[0],frames[1],frames[2],int(horizon))); targets.append(target(frames[0],frames[int(horizon)]))
    raw_context=torch.stack(contexts); raw_target=torch.stack(targets)
    return {"context_mean":raw_context.mean(0),"context_std":raw_context.std(0).clamp_min(1e-12),"target_mean":raw_target.mean(0),"target_std":raw_target.std(0).clamp_min(1e-12)}

def normalized_context(initial, one, two, horizons, normalizers):
    values=torch.stack([context(initial,one,two,int(horizon)) for horizon in horizons])
    return (values-normalizers["context_mean"])/normalizers["context_std"]

def positive_yz_box(initial, standardized_displacement, normalizers):
    displacement=standardized_displacement*normalizers["target_std"]+normalizers["target_mean"]
    # For physical box lengths (>5 in this cohort), beta=10 softplus is
    # numerically the identity while preventing invalid negative extrapolation.
    return torch.nn.functional.softplus(initial["box_tensor"][1:]+displacement,beta=10.0)+1e-6

def raw_prediction(model, graph, initial, one, two, horizons, normalizers):
    output=model.decode(model.encode(*graph),normalized_context(initial,one,two,horizons,normalizers))
    return positive_yz_box(initial,output,normalizers),output

def affine_prediction(initial, one, two, horizon, yz_box):
    x,_=prescribed_x(initial,one,two,horizon); box=torch.cat((x.reshape(1),yz_box))
    if torch.any(box<=0): raise ValueError("Positive box parameterization failed.")
    return base.fractional_coordinates(initial["pos"],initial["box_tensor"])*box,box

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--variant",choices=("static_mean","static_token"),required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    protocol=json.loads((HERE/"protocol.json").read_text()); split_path=(HERE/protocol["split_manifest"]).resolve(); split=json.loads(split_path.read_text()); all_ids=[i for key in ("train","validation","final_test") for i in split[key]]
    if len(split["train"])!=48 or len(split["validation"])!=12 or len(split["final_test"])!=12 or len(set(all_ids))!=72: raise ValueError("Expected frozen disjoint v2 48/12/12 split.")
    if args.seed!=123: raise ValueError("Frozen screen permits seed 123 only.")
    if args.smoke: train_ids,validation_ids,evaluation_ids=split["train"][:4],split["validation"][:1],split["validation"][1:2]; epochs=1; per_graph=2; evaluation_label="smoke_validation_1"
    else: train_ids,validation_ids,evaluation_ids=split["train"],split["validation"],split["validation"]; epochs=protocol["training"]["max_epochs"]; per_graph=protocol["training"]["target_frames_per_train_graph_epoch"]; evaluation_label="validation"
    if set(evaluation_ids)-set(split["validation"]): raise RuntimeError("Validation-only harness must never load final_test.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); fit_ids=list(dict.fromkeys(train_ids+validation_ids)); loaded={i:base.frame_data(base.load_cold_trajectory(i)) for i in fit_ids}
    normalizers=fit_normalizers([loaded[i] for i in train_ids],protocol["normalization"]["seed"],protocol["normalization"]["centers_per_trajectory"]); model=StaticGlobalBoxResponse(pooling=protocol["variants"][args.variant]["pooling"]); optimizer=torch.optim.Adam(model.parameters(),lr=protocol["training"]["learning_rate"],weight_decay=protocol["training"]["weight_decay"]); rng=np.random.default_rng(args.seed); history=[]; best,best_state,stale=float("inf"),None,0
    graphs={i:reference_graph(loaded[i][0],protocol["contact_cutoff"]) for i in fit_ids}
    for epoch in range(epochs):
        model.train(); losses=[]
        for identifier in train_ids:
            frames=loaded[identifier]; horizons=np.arange(3,400) if per_graph==397 else rng.choice(np.arange(3,400),size=per_graph,replace=False); normalized=normalized_context(frames[0],frames[1],frames[2],horizons,normalizers); embedding=model.encode(*graphs[identifier]); standardized=model.decode(embedding,normalized); prediction=positive_yz_box(frames[0],standardized,normalizers); truths=torch.stack([frames[int(h)]["box_tensor"][1:] for h in horizons]); loss=((prediction-truths)/normalizers["target_std"]).square().mean(); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["training"]["gradient_clip_norm"]); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation=[]
            for identifier in validation_ids:
                frames=loaded[identifier]; horizons=protocol["training"]["selection_horizons"]; predicted,_=raw_prediction(model,graphs[identifier],frames[0],frames[1],frames[2],horizons,normalizers); truth=torch.stack([frames[h]["box_tensor"][1:] for h in horizons]); validation.append((predicted-truth).square().mean().item())
            value=float(np.mean(validation))
        history.append({"epoch":epoch+1,"training_standardized_total_yz_box_state_mse":float(np.mean(losses)),"validation_raw_yz_box_length_mse":value,"selection_horizons":protocol["training"]["selection_horizons"]})
        if value<best: best,best_state,stale=value,{n:v.detach().clone() for n,v in model.state_dict().items()},0
        else: stale+=1
        if epoch+1>=protocol["training"]["minimum_epochs"] and stale>=protocol["training"]["patience"]: break
    if best_state is None: raise RuntimeError("No state-selected checkpoint.")
    model.load_state_dict(best_state); args.output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best_state,"normalizers":normalizers},args.output/"checkpoint.pt")
    with (args.output/"training_history.csv").open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    rows=[]; model.eval()
    with torch.no_grad():
        for identifier in evaluation_ids:
            frames=loaded.get(identifier) or base.frame_data(base.load_cold_trajectory(identifier)); graph=reference_graph(frames[0],protocol["contact_cutoff"])
            horizons=list(range(3,400)); embedding=model.encode(*graph); normalized=normalized_context(frames[0],frames[1],frames[2],horizons,normalizers); standardized=model.decode(embedding,normalized); yz_boxes=positive_yz_box(frames[0],standardized,normalizers)
            for horizon,yz_box in zip(horizons,yz_boxes):
                positions,box=affine_prediction(frames[0],frames[1],frames[2],horizon,yz_box); py,pz=base.aligned_p_ratios(frames[0]["pos"],positions,box); ty,tz=base.aligned_p_ratios(frames[0]["pos"],frames[horizon]["pos"],frames[horizon]["box_tensor"])
                rows.append({"network_id":identifier,"frame":horizon,"position_mse_physical_minimum_image":base.minimum_image(positions-frames[horizon]["pos"],frames[horizon]["box_tensor"]).square().mean().item(),"yz_box_length_mse":(box[1:]-frames[horizon]["box_tensor"][1:]).square().mean().item(),"pred_p_y_over_x":py,"true_p_y_over_x":ty,"pred_p_z_over_x":pz,"true_p_z_over_x":tz})
    with (args.output/"per_network_per_frame.csv").open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={"variant":args.variant,"seed":args.seed,"smoke":args.smoke,"evaluation_split":evaluation_label,"best_validation_raw_yz_box_length_mse":best,"epochs_completed":epoch+1,"test":{}}
    for horizon in protocol["evaluation"]["horizons"]:
        selected=[row for row in rows if row["frame"]==horizon]; yr2,yvalid=base.r2(selected,"true_p_y_over_x","pred_p_y_over_x"); zr2,zvalid=base.r2(selected,"true_p_z_over_x","pred_p_z_over_x"); summary["test"][str(horizon)]={"position_mse_physical_minimum_image":float(np.mean([row["position_mse_physical_minimum_image"] for row in selected])),"yz_box_length_mse":float(np.mean([row["yz_box_length_mse"] for row in selected])),"p_ratio_r2_y_over_x":yr2,"p_ratio_r2_z_over_x":zr2,"p_ratio_y_valid_total":[yvalid,len(selected)],"p_ratio_z_valid_total":[zvalid,len(selected)]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    data_ids=list(dict.fromkeys(fit_ids+evaluation_ids)); recipe={"runner_sha256":sha256(Path(__file__)),"base_v2_runner_sha256":sha256(BASE_PATH),"protocol_sha256":sha256(HERE/"protocol.json"),"split_manifest_sha256":sha256(split_path),"model_source_sha256":sha256(ROOT/"src/lss/dynamics/static_response_head.py"),"fractional_model_source_sha256":sha256(ROOT/"src/lss/dynamics/fractional_contact_simulator.py"),"loader_sha256":sha256(ROOT/"src/lss/dynamics/cold_graphs.py"),"cohort_sha256":sha256(base.COLD_GRAPH_ROOT/"clean_400.json"),"trajectory_sha256":{i:sha256(base.COLD_GRAPH_ROOT/f"{i}.pt") for i in data_ids},"variant":args.variant,"contact_cutoff":protocol["contact_cutoff"],"seed":args.seed,"smoke":args.smoke,"normalizer":{"train_ids":train_ids,**protocol["normalization"]},"train_ids":train_ids,"validation_ids":validation_ids,"evaluation_split":evaluation_label,"evaluation_ids":evaluation_ids}
    (args.output/"recipe.json").write_text(json.dumps(recipe,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
