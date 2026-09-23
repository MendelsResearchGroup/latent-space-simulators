import argparse,json,subprocess,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RESULTS=ROOT/"notebooks/results/graphs_cold_global_v8"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--smoke-retry1",action="store_true");p.add_argument("--production",action="store_true");a=p.parse_args()
 if sum((a.smoke,a.smoke_retry1,a.production))!=1:raise ValueError("choose exactly one")
 kind="production" if a.production else ("smoke_retry1" if a.smoke_retry1 else "smoke");ledger=RESULTS/f"{kind}_jobs.json";RESULTS.mkdir(parents=True,exist_ok=True)
 if ledger.exists():raise FileExistsError(ledger)
 if a.production:
  s=RESULTS/"smoke_retry1"/"123"/"summary.json"
  if not s.is_file() or json.loads(s.read_text()).get("evaluation_split")!="smoke_validation_1":raise RuntimeError("smoke required")
 out=RESULTS/kind/"123";out.mkdir(parents=True,exist_ok=True);job=subprocess.check_output(["qsub","-N",f"coldv8_{kind}","-v",f"LSS_SEED=123,LSS_SMOKE={int(not a.production)},LSS_OUTPUT={out}",str(HERE/"run.pbs")],text=True).strip();ledger.write_text(json.dumps({"jobs":{"node_only_global":job},"runner_sha256":sha(HERE/"run.py"),"protocol_sha256":sha(HERE/"protocol.json"),"split_manifest_sha256":sha((HERE/"../graphs_cold_global/split_manifest_v2.json").resolve()),"model_source_sha256":sha(ROOT/"src/lss/dynamics/node_only_global_simulator.py")})+"\n");print(job)
if __name__=="__main__":main()
