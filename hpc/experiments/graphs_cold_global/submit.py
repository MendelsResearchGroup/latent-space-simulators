"""Submit only smoke jobs; production is deliberately gated on recorded smoke success."""
from __future__ import annotations
import argparse, subprocess, json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
RESULTS=HERE.parents[2]/"notebooks/results/graphs_cold_global"
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--smoke",action="store_true"); p.add_argument("--production",action="store_true"); a=p.parse_args()
    if a.smoke == a.production: raise ValueError("Choose exactly one of --smoke or --production.")
    root=RESULTS; root.mkdir(parents=True,exist_ok=True)
    if a.production:
        required=[root/"smoke"/model/"123"/"summary.json" for model in ("local","global")]
        if not all(x.is_file() for x in required): raise RuntimeError("Production is forbidden until both smoke summaries exist.")
        jobs={}
        for model in ("local","global"):
            for seed in (123,456,786):
                out=root/"production"/model/str(seed); out.mkdir(parents=True,exist_ok=True)
                command=["qsub","-N",f"cold_{model}_{seed}","-v",f"LSS_MODEL={model},LSS_SEED={seed},LSS_SMOKE=0,LSS_OUTPUT={out}",str(HERE/"run.pbs")]
                jobs[f"{model}_{seed}"]=subprocess.check_output(command,text=True).strip()
        (root/"production_jobs.json").write_text(json.dumps({"jobs":jobs,"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha(HERE/"split_manifest.json"),"runner_sha256":sha(HERE/"run.py")},indent=2,sort_keys=True)+"\n")
        return
    jobs={}
    for model in ("local","global"):
        out=root/"smoke"/model/"123"; out.mkdir(parents=True,exist_ok=True)
        command=["qsub","-N",f"cold_smoke_{model}","-v",f"LSS_MODEL={model},LSS_SEED=123,LSS_SMOKE=1,LSS_OUTPUT={out}",str(HERE/"run.pbs")]
        job_id=subprocess.check_output(command,text=True).strip(); jobs[model]=job_id; print(job_id)
    (root/"smoke_jobs.json").write_text(json.dumps({"jobs":jobs,"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha(HERE/"split_manifest.json"),"runner_sha256":sha(HERE/"run.py")},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
