"""Submit v4 only after the two validation-only smoke artifacts pass the gate."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; RESULTS=ROOT/"notebooks/results/graphs_cold_global_v4"; VARIANTS=("static_mean","static_token")
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def record(jobs,kind): return {"kind":kind,"jobs":jobs,"protocol_sha256":sha256(HERE/"protocol.json"),"split_manifest_sha256":sha256((HERE/"../graphs_cold_global/split_manifest_v2.json").resolve()),"runner_sha256":sha256(HERE/"run.py"),"model_source_sha256":sha256(ROOT/"src/lss/dynamics/static_response_head.py"),"base_v2_runner_sha256":sha256(HERE.parent/"graphs_cold_global_v2/run.py")}
def smoke_passed(path):
    return path.is_file() and json.loads(path.read_text()).get("smoke") is True and json.loads(path.read_text()).get("evaluation_split")=="smoke_validation_1"
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--production",action="store_true"); args=parser.parse_args()
    if args.smoke==args.production: raise ValueError("Choose exactly one of --smoke or --production.")
    RESULTS.mkdir(parents=True,exist_ok=True); jobs={}
    if args.production:
        required=[RESULTS/"smoke_retry3"/v/"123"/"summary.json" for v in VARIANTS]
        if not all(smoke_passed(p) for p in required): raise RuntimeError("Production is forbidden until both valid smoke summaries exist.")
        for v in VARIANTS:
            out=RESULTS/"production_retry1"/v/"123"; out.mkdir(parents=True,exist_ok=True); jobs[v]=subprocess.check_output(["qsub","-N",f"coldv4_r1_{v}","-v",f"LSS_VARIANT={v},LSS_SEED=123,LSS_SMOKE=0,LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip()
        (RESULTS/"production_retry1_jobs.json").write_text(json.dumps(record(jobs,"production_retry1"),indent=2,sort_keys=True)+"\n")
    else:
        for v in VARIANTS:
            out=RESULTS/"smoke_retry3"/v/"123"; out.mkdir(parents=True,exist_ok=True); jobs[v]=subprocess.check_output(["qsub","-N",f"coldv4_smoke_r3_{v}","-v",f"LSS_VARIANT={v},LSS_SEED=123,LSS_SMOKE=1,LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip()
        (RESULTS/"smoke_retry3_jobs.json").write_text(json.dumps(record(jobs,"smoke_retry3"),indent=2,sort_keys=True)+"\n")
    for job in jobs.values(): print(job)
if __name__=="__main__": main()
