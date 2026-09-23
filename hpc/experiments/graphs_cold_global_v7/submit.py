from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; RESULTS=ROOT/"notebooks/results/graphs_cold_global_v7"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ok(p): return p.is_file() and json.loads(p.read_text()).get("smoke") is True and json.loads(p.read_text()).get("evaluation_split")=="smoke_validation_1"
def main():
 a=argparse.ArgumentParser(); a.add_argument("--smoke",action="store_true"); a.add_argument("--production",action="store_true"); x=a.parse_args()
 if x.smoke==x.production: raise ValueError("Choose exactly one of --smoke or --production.")
 kind="smoke" if x.smoke else "production"; ledger=RESULTS/f"{kind}_jobs.json"; RESULTS.mkdir(parents=True,exist_ok=True)
 if ledger.exists(): raise FileExistsError(ledger)
 if x.production and not ok(RESULTS/"smoke"/"123"/"summary.json"): raise RuntimeError("Production requires valid smoke.")
 out=RESULTS/kind/"123"; out.mkdir(parents=True,exist_ok=True); job=subprocess.check_output(["qsub","-N",f"coldv7_{kind}","-v",f"LSS_SEED=123,LSS_SMOKE={int(x.smoke)},LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip(); ledger.write_text(json.dumps({"kind":kind,"jobs":{"initial_motion_token_residual":job},"runner_sha256":sha(HERE/"run.py"),"protocol_sha256":sha(HERE/"protocol.json"),"model_source_sha256":sha(ROOT/"src/lss/dynamics/initial_motion_response.py")},indent=2)+"\n"); print(job)
if __name__=="__main__": main()
