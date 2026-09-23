"""Validation-only v5 residual screen with deterministic graph-trajectory accumulation."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, random
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; V5_PATH=ROOT/"hpc/experiments/graphs_cold_global_v5/run.py"
spec=importlib.util.spec_from_file_location("cold_v5_base",V5_PATH); v5=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(v5)
base=v5.base; v4=v5.v4; StaticGlobalBoxResponse=v5.StaticGlobalBoxResponse
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def shuffled_batches(ids, rng, batch_size):
    order=list(ids); rng.shuffle(order); return [order[start:start+batch_size] for start in range(0,len(order),batch_size)]
def scaled_graph_loss(loss, actual_batch_size):
    if actual_batch_size < 1: raise ValueError("actual_batch_size must be positive.")
    return loss/actual_batch_size
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    protocol=json.loads((HERE/"protocol.json").read_text()); split_path=(HERE/protocol["split_manifest"]).resolve(); split=json.loads(split_path.read_text()); all_ids=[i for key in ("train","validation","final_test") for i in split[key]]
    if len(split["train"])!=48 or len(split["validation"])!=12 or len(split["final_test"])!=12 or len(set(all_ids))!=72: raise ValueError("Expected frozen disjoint v2 48/12/12 split.")
    if args.seed!=123: raise ValueError("Frozen screen permits seed 123 only.")
    if args.smoke: train_ids,validation_ids,evaluation_ids=split["train"][:4],split["validation"][:1],split["validation"][1:2]; epochs=1; per_graph=2; label="smoke_validation_1"
    else: train_ids,validation_ids,evaluation_ids=split["train"],split["validation"],split["validation"]; epochs=protocol["training"]["max_epochs"]; per_graph=protocol["training"]["target_frames_per_train_graph_epoch"]; label="validation"
    v5.assert_validation_boundary(split,evaluation_ids)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); fit_ids=list(dict.fromkeys(train_ids+validation_ids)); loaded={i:base.frame_data(base.load_cold_trajectory(i)) for i in fit_ids}; normalizers=v5.fit_ols_and_normalizers([loaded[i] for i in train_ids]); beta=protocol["softplus_beta"]
    model=StaticGlobalBoxResponse(pooling="token"); torch.nn.init.zeros_(model.decoder[-1].weight); torch.nn.init.zeros_(model.decoder[-1].bias); optimizer=torch.optim.Adam(model.parameters(),lr=protocol["training"]["learning_rate"],weight_decay=protocol["training"]["weight_decay"]); graphs={i:v4.reference_graph(loaded[i][0],protocol["contact_cutoff"]) for i in fit_ids}; rng=np.random.default_rng(args.seed); best,best_state,stale=float("inf"),None,0; history=[]; total_steps=0
    for epoch in range(epochs):
        model.train(); losses=[]; steps_this_epoch=0
        for batch in shuffled_batches(train_ids,rng,protocol["training"]["graph_trajectory_gradient_accumulation"]):
            optimizer.zero_grad()
            for identifier in batch:
                frames=loaded[identifier]; horizons=np.arange(3,400) if per_graph==397 else rng.choice(np.arange(3,400),size=per_graph,replace=False); predicted,_,_,_=v5.predict(model,graphs[identifier],frames,horizons,normalizers,beta); truth=frames[0]["box_tensor"][1:]+v5.raw_targets(frames,horizons); loss=((predicted-truth)/normalizers["residual_std"]).square().mean(); scaled_graph_loss(loss,len(batch)).backward(); losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),protocol["training"]["gradient_clip_norm"]); optimizer.step(); steps_this_epoch+=1; total_steps+=1
        model.eval()
        with torch.no_grad():
            values=[]
            for identifier in validation_ids:
                frames=loaded[identifier]; horizons=protocol["training"]["selection_horizons"]; predicted,_,_,_=v5.predict(model,graphs[identifier],frames,horizons,normalizers,beta); truth=frames[0]["box_tensor"][1:]+v5.raw_targets(frames,horizons); values.append((predicted-truth).square().mean().item())
            value=float(np.mean(values))
        history.append({"epoch":epoch+1,"training_residual_scaled_yz_box_mse":float(np.mean(losses)),"validation_raw_yz_box_length_mse":value,"optimizer_steps_this_epoch":steps_this_epoch,"optimizer_steps_cumulative":total_steps,"selection_horizons":protocol["training"]["selection_horizons"]})
        if value<best: best,best_state,stale=value,{n:v.detach().clone() for n,v in model.state_dict().items()},0
        else: stale+=1
        if epoch+1>=protocol["training"]["min_epochs"] and stale>=protocol["training"]["patience"]: break
    if best_state is None: raise RuntimeError("No state-selected checkpoint.")
    model.load_state_dict(best_state); args.output.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best_state,"normalizers":normalizers},args.output/"checkpoint.pt")
    with (args.output/"training_history.csv").open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    rows=[]; ols_rows=[]; model.eval()
    with torch.no_grad():
        for identifier in evaluation_ids:
            frames=loaded.get(identifier) or base.frame_data(base.load_cold_trajectory(identifier)); graph=v4.reference_graph(frames[0],protocol["contact_cutoff"]); embedding=model.encode(*graph); horizons=list(range(3,400)); predicted_boxes,ols_boxes,_,_=v5.predict(model,graph,frames,horizons,normalizers,beta,embedding)
            for horizon,predicted_box,ols_box in zip(horizons,predicted_boxes,ols_boxes):
                for output,kind in ((predicted_box,"model"),(ols_box,"ols")):
                    positions,box=v5.affine_from_box(frames,horizon,output); py,pz=base.aligned_p_ratios(frames[0]["pos"],positions,box); ty,tz=base.aligned_p_ratios(frames[0]["pos"],frames[horizon]["pos"],frames[horizon]["box_tensor"]); row={"network_id":identifier,"frame":horizon,"position_mse_physical_minimum_image":base.minimum_image(positions-frames[horizon]["pos"],frames[horizon]["box_tensor"]).square().mean().item(),"yz_box_length_mse":(output-frames[horizon]["box_tensor"][1:]).square().mean().item(),"pred_p_y_over_x":py,"true_p_y_over_x":ty,"pred_p_z_over_x":pz,"true_p_z_over_x":tz}; (rows if kind=="model" else ols_rows).append(row)
    for name,values in (("per_network_per_frame.csv",rows),("ols_per_network_per_frame.csv",ols_rows)):
        with (args.output/name).open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    summary={"variant":protocol["variant"],"seed":args.seed,"smoke":args.smoke,"evaluation_split":label,"best_validation_raw_yz_box_length_mse":best,"epochs_completed":epoch+1,"optimizer_steps_total":total_steps,"test":{}}
    for horizon in protocol["evaluation"]["horizons"]: summary["test"][str(horizon)]={**v5.metric_rows([r for r in rows if r["frame"]==horizon]),"ols_control":v5.metric_rows([r for r in ols_rows if r["frame"]==horizon])}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    serial={key:value.tolist() for key,value in normalizers.items()}; data_ids=list(dict.fromkeys(fit_ids+evaluation_ids)); recipe={"runner_sha256":sha256(Path(__file__)),"base_v5_runner_sha256":sha256(V5_PATH),"base_v4_runner_sha256":sha256(v5.V4_PATH),"base_v2_runner_sha256":sha256(v4.BASE_PATH),"protocol_sha256":sha256(HERE/"protocol.json"),"split_manifest_sha256":sha256(split_path),"model_source_sha256":sha256(ROOT/"src/lss/dynamics/static_response_head.py"),"loader_sha256":sha256(ROOT/"src/lss/dynamics/cold_graphs.py"),"cohort_sha256":sha256(base.COLD_GRAPH_ROOT/"clean_400.json"),"trajectory_sha256":{i:sha256(base.COLD_GRAPH_ROOT/f"{i}.pt") for i in data_ids},"seed":args.seed,"smoke":args.smoke,"ols_and_normalizers":serial,"train_ids":train_ids,"validation_ids":validation_ids,"evaluation_split":label,"evaluation_ids":evaluation_ids}
    (args.output/"recipe.json").write_text(json.dumps(recipe,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
