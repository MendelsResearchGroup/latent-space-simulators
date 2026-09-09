#!/usr/bin/env python3
"""Fit one frozen-AE compact-LJ propagator, then perform post-fit evaluation."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys, time
from pathlib import Path

ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path.cwd())).resolve()
CODE = Path(os.environ.get("LSS_CODE_ROOT", ROOT)).resolve()
if not (CODE / "src" / "lss").is_dir():
    raise RuntimeError(f"LSS_CODE_ROOT lacks frozen src/lss: {CODE}")
sys.path.insert(0, str(CODE / "src"))
import pandas as pd
import torch
from lss.dynamics.experiment import run_latent_experiment
from evaluate import FRAMES, assert_d4_reconstruction_parity, evaluate

OUTROOT = ROOT / "notebooks/results/compact_lj_response_rollout/runs"
SEEDS = (3456456, 123, 456, 786, 2026)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(8*1024*1024), b""): h.update(x)
    return h.hexdigest()

def tensor_digest(state):
    h=hashlib.sha256()
    for key in sorted(state):
        value=state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()

def verify_frozen_source():
    manifest=CODE/"hashes.json"
    if not manifest.exists(): raise ValueError("Frozen source lacks hashes.json.")
    recorded=json.loads(manifest.read_text())
    actual={key:sha(CODE/key) for key in recorded}
    if actual != recorded: raise ValueError("Frozen LSS source hash manifest mismatch.")
    return sha(manifest)

def paths(variant, seed):
    if variant == "mp2_r16_d4":
        base=ROOT/"notebooks/results/compact_lj_reconstruction"/f"{variant}_s{seed}_code_v2"
    else:
        base=ROOT/"notebooks/results/compact_lj_mp2d"/f"{variant}_s{seed}_code_v4"
    return base/"ae.pt", base/"recipe.json", base/"completed.json"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant", choices=("mp2_r16_d2","mp2_r16_d4"), required=True)
    p.add_argument("--seed", choices=SEEDS, type=int, required=True); p.add_argument("--strategy", choices=("one_step","multistep8"), required=True)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--tag",default=""); a=p.parse_args(); start=time.time()
    random.seed(a.seed); np_seed = __import__("numpy"); np_seed.random.seed(a.seed); torch.manual_seed(a.seed)
    checkpoint, recipe_file, complete_file = paths(a.variant,a.seed)
    out=OUTROOT / (f"{a.variant}_{a.strategy}_s{a.seed}" + ("_smoke" if a.smoke else "") + (f"_{a.tag}" if a.tag else ""))
    out.mkdir(parents=True, exist_ok=False)
    try:
        source_manifest_sha=verify_frozen_source()
        recipe=json.loads(recipe_file.read_text()); done=json.loads(complete_file.read_text())
        ck_sha, recipe_sha=sha(checkpoint),sha(recipe_file)
        if done.get("ae_file_sha256") != ck_sha or recipe["response_selection"] or recipe["final_test_used"]:
            raise ValueError("Frozen AE provenance or response/final-test contract failed.")
        frozen_bundle=torch.load(checkpoint,map_location="cpu",weights_only=False)
        original_state=frozen_bundle["ae_state_dict"]
        original_stats=frozen_bundle.get("stats",frozen_bundle.get("normalizers"))
        if original_stats is None: raise ValueError("Frozen AE bundle lacks normalizers.")
        cfg=dict(recipe["config"]); cfg.update({
            "should_train_propagator": True, "should_rollout": False, "force_train": True,
            "pretrained_ae_cache_path": str(checkpoint),
            "pretrained_ae_config_keys": ["autoencoder_model","latent_dim","latent_tokens","hidden_size","edge_feature_dim","message_passing_steps","node_feature_mode","ae_target_mode","edge_mode","pos_dim"],
            "pretrained_ae_require_matching_config": True, "pretrained_ae_skip_stat_fitting": True,
            "pretrained_ae_require_matching_normalizers": False,
            "cache_path":str(out / "propagator_bundle.pt"), "cache_require_matching_config":True,
            "propagator_model": "delta_mlp", "propagator_loss": "delta",
            "propagator_objective": "one_step" if a.strategy=="one_step" else "multistep",
            "propagator_multistep_horizons": list(range(1,9)), "propagator_hidden_size":64,
            "propagator_use_static_context":True, "propagator_context_pool":"mean",
            "graph_context_dim":16,
            "propagator_context_include_temperature":False, "propagator_context_include_source_id":False,
            "propagator_source_loss_reduction":"equal", "propagator_position_loss_weight":0.0,
            "physics_loss_enabled":False, "propagator_checkpoint_metric":None,
            "propagator_checkpoint_mode":"min", "propagator_rollout_eval_every_epoch":False,
            "dyn_max_epochs": 1 if a.smoke else 30, "dyn_patience":1 if a.smoke else 6,
            "dyn_lr":1e-4, "dyn_weight_decay":1e-5, "batch_graphs":32,
            "dyn_max_train_transitions_per_sim": 12 if a.smoke else 199,
            "dyn_max_val_transitions_per_sim": 12 if a.smoke else 199,
            "rollout_steps_grid":list(FRAMES), "rollout_eval_splits":[], "model_seed":a.seed,
        })
        planned={"variant":a.variant,"strategy":a.strategy,"seed":a.seed,"smoke":a.smoke,
            "checkpoint":str(checkpoint),"checkpoint_sha256":ck_sha,"ae_recipe":str(recipe_file),
            "ae_recipe_sha256":recipe_sha,"runner_sha256":sha(Path(__file__)),"code_root":str(CODE),"code_hash_manifest_sha256":source_manifest_sha,
            "frames":FRAMES,"selection":"validation latent delta loss only","response_selection":False,
            "final_test_used":False,"config":cfg,"source":recipe["source"],"mixed_evaluation":recipe["mixed_evaluation"]}
        (out/"recipe.json").write_text(json.dumps(planned,indent=2)+"\n")
        result=run_latent_experiment(recipe["source"],cfg,device="cpu")
        if tensor_digest(result["ae"].state_dict()) != tensor_digest(original_state):
            raise ValueError("In-memory AE state changed from its frozen checkpoint.")
        for key in ("target_mean","target_std","node_feature_mean","node_feature_std","edge_mean","edge_std","ref_edge_mean","ref_edge_std"):
            saved=original_stats.get(key,original_stats.get({"ref_edge_mean":"edge_mean","ref_edge_std":"edge_std"}.get(key,key)))
            if saved is None or not torch.equal(result["normalizers"][key].detach().cpu(),saved.detach().cpu()):
                raise ValueError(f"Frozen normalizer mismatch: {key}")
        table=evaluate(result,recipe,variant=a.variant,strategy=a.strategy,seed=a.seed,checkpoint_sha256=ck_sha)
        expected=4*20*len(FRAMES)*2
        if len(table)!=expected or set(table.kind)!={"reconstruction","autonomous"}:
            raise ValueError(f"Unexpected evaluation rows {len(table)} != {expected}.")
        parity = assert_d4_reconstruction_parity(table, seed=a.seed) if a.variant == "mp2_r16_d4" else None
        table.to_csv(out/"per_network_rows.csv",index=False)
        result["dyn_history"].to_csv(out/"dynamics_history.csv",index=False)
        planned.update({"checkpoint_sha256_after":sha(checkpoint),"frozen_ae_state_sha256":tensor_digest(original_state),
                        "frozen_ae_normalizers_verified":True,"propagator_bundle":str(out/"propagator_bundle.pt")})
        if parity is not None: planned["reconstruction_parity"] = parity
        (out/"recipe.json").write_text(json.dumps(planned,indent=2)+"\n")
        (out/"completed.json").write_text(json.dumps({"status":"completed","rows":len(table),"job_id":os.environ.get("PBS_JOBID"),"seconds":time.time()-start,
            "checkpoint_sha256":ck_sha,"checkpoint_sha256_after":sha(checkpoint)},indent=2)+"\n")
    except Exception as e:
        (out/"failed.json").write_text(json.dumps({"status":"failed","error":f"{type(e).__name__}: {e}","job_id":os.environ.get("PBS_JOBID")},indent=2)+"\n"); raise
if __name__=="__main__": main()
