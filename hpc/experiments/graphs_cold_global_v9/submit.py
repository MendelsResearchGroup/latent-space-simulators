import argparse,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RESULTS=ROOT/"notebooks/results/graphs_cold_global_v9"
def main():
 p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--production",action="store_true");a=p.parse_args()
 if a.smoke==a.production:raise ValueError("choose one")
 kind="smoke" if a.smoke else "production";ledger=RESULTS/f"{kind}_jobs.json";RESULTS.mkdir(parents=True,exist_ok=True)
 if ledger.exists():raise FileExistsError(ledger)
 if a.production:
  x=RESULTS/"smoke"/"123"/"summary.json"
  if not x.is_file() or json.loads(x.read_text()).get("evaluation_split")!="smoke_validation_1":raise RuntimeError("smoke required")
 out=RESULTS/kind/"123";out.mkdir(parents=True,exist_ok=True);job=subprocess.check_output(["qsub","-N",f"coldv9_{kind}","-v",f"LSS_SEED=123,LSS_SMOKE={int(a.smoke)},LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip();ledger.write_text(json.dumps({"jobs":{"v9_phase1":job}})+"\n");print(job)
if __name__=="__main__":main()
