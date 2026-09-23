"""Submit the frozen validation-only screen after its smoke gate passes."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "notebooks/results/graphs_cold_global_v2"
VARIANTS = ("fractional_bond", "fractional_contact")

def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def record(jobs, kind):
    return {"kind": kind, "jobs": jobs, "protocol_sha256": sha256(HERE / "protocol.json"), "split_manifest_sha256": sha256((HERE / "../graphs_cold_global/split_manifest_v2.json").resolve()), "runner_sha256": sha256(HERE / "run.py")}
def smoke_passed(path):
    if not path.is_file(): return False
    summary = json.loads(path.read_text())
    return summary.get("smoke") is True and summary.get("evaluation_split") == "smoke_validation_1"
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--production", action="store_true"); args = parser.parse_args()
    if args.smoke == args.production: raise ValueError("Choose exactly one of --smoke or --production.")
    RESULTS.mkdir(parents=True, exist_ok=True); jobs = {}
    if args.production:
        required = [RESULTS / "smoke" / variant / "123" / "summary.json" for variant in VARIANTS]
        if not all(smoke_passed(path) for path in required): raise RuntimeError("Production is forbidden until both valid smoke summaries exist.")
        for variant in VARIANTS:
            output = RESULTS / "production" / variant / "123"; output.mkdir(parents=True, exist_ok=True)
            jobs[variant] = subprocess.check_output(["qsub", "-N", f"coldv2_{variant}", "-v", f"LSS_VARIANT={variant},LSS_SEED=123,LSS_SMOKE=0,LSS_OUTPUT={output}", str(HERE / "run.pbs")], text=True).strip()
        (RESULTS / "production_jobs.json").write_text(json.dumps(record(jobs, "production"), indent=2, sort_keys=True) + "\n")
    else:
        for variant in VARIANTS:
            output = RESULTS / "smoke" / variant / "123"; output.mkdir(parents=True, exist_ok=True)
            jobs[variant] = subprocess.check_output(["qsub", "-N", f"coldv2_smoke_{variant}", "-v", f"LSS_VARIANT={variant},LSS_SEED=123,LSS_SMOKE=1,LSS_OUTPUT={output}", str(HERE / "run.pbs")], text=True).strip()
        (RESULTS / "smoke_jobs.json").write_text(json.dumps(record(jobs, "smoke"), indent=2, sort_keys=True) + "\n")
    for job in jobs.values(): print(job)
if __name__ == "__main__": main()
