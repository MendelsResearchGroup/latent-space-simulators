"""Submit v5 only after its validation-only smoke artifact passes."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; RESULTS=ROOT/"notebooks/results/graphs_cold_global_v5"
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def record(jobs,kind): return {"kind":kind,"jobs":jobs,"protocol_sha256":sha256(HERE/"protocol.json"),"split_manifest_sha256":sha256((HERE/"../graphs_cold_global/split_manifest_v2.json").resolve()),"runner_sha256":sha256(HERE/"run.py"),"base_v4_runner_sha256":sha256(HERE.parent/"graphs_cold_global_v4/run.py"),"model_source_sha256":sha256(ROOT/"src/lss/dynamics/static_response_head.py")}
def smoke_passed(path): return path.is_file() and json.loads(path.read_text()).get("smoke") is True and json.loads(path.read_text()).get("evaluation_split")=="smoke_validation_1"
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--production",action="store_true"); args=parser.parse_args()
    if args.smoke==args.production: raise ValueError("Choose exactly one of --smoke or --production.")
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.production:
        if not smoke_passed(RESULTS/"smoke"/"123"/"summary.json"): raise RuntimeError("Production is forbidden until the valid smoke summary exists.")
        out=RESULTS/"production"/"123"; out.mkdir(parents=True,exist_ok=True); job=subprocess.check_output(["qsub","-N","coldv5_residual","-v",f"LSS_SEED=123,LSS_SMOKE=0,LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip(); (RESULTS/"production_jobs.json").write_text(json.dumps(record({"static_token_residual":job},"production"),indent=2,sort_keys=True)+"\n")
    else:
        out=RESULTS/"smoke"/"123"; out.mkdir(parents=True,exist_ok=True); job=subprocess.check_output(["qsub","-N","coldv5_smoke","-v",f"LSS_SEED=123,LSS_SMOKE=1,LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip(); (RESULTS/"smoke_jobs.json").write_text(json.dumps(record({"static_token_residual":job},"smoke"),indent=2,sort_keys=True)+"\n")
    print(job)
if __name__=="__main__": main()
