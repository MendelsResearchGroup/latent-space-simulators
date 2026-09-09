#!/usr/bin/env python3
"""Submit smoke, then the twenty state-selected compact-LJ rollout cells."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"notebooks/results/compact_lj_response_rollout"
HERE=Path(__file__).resolve().parent
SEEDS=(3456456,123,456,786,2026); VARIANTS=("mp2_r16_d2","mp2_r16_d4"); STRATEGIES=("one_step","multistep8")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
 return h.hexdigest()
def append(row):
 with (OUT/"jobs.jsonl").open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
def qsub(args): return subprocess.check_output(["qsub",*args],text=True).strip().splitlines()[-1]
def now(): return datetime.now(timezone.utc).isoformat()
def ae_path(v,s): return ROOT/("notebooks/results/compact_lj_reconstruction" if v.endswith("d4") else "notebooks/results/compact_lj_mp2d")/f"{v}_s{s}_code_v{'2' if v.endswith('d4') else '4'}"/"ae.pt"
def frozen():
 d=OUT/"code_rollout_v4"; manifest=d/"hashes.json"
 if d.exists():
  recorded=json.loads(manifest.read_text()); actual={p.name:sha(p) for p in d.glob("*.py")} | {"run.pbs":sha(d/"run.pbs")}
  if actual!=recorded["files"]: raise RuntimeError("Frozen rollout harness hash mismatch.")
  return d
 d.mkdir(parents=True)
 for n in ("run.py","evaluate.py","run.pbs","submit.py"): shutil.copy2(HERE/n,d/n)
 files={p.name:sha(p) for p in d.glob("*.py")} | {"run.pbs":sha(d/"run.pbs")}
 manifest.write_text(json.dumps({"files":files,"frozen_at":now()},indent=2)+"\n")
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument("--smoke",action="store_true");p.add_argument("--submit",action="store_true");a=p.parse_args()
 if a.smoke==a.submit: raise SystemExit("Choose exactly one of --smoke or --submit.")
 OUT.mkdir(parents=True,exist_ok=True); ledger=OUT/"jobs.jsonl"; old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
 snap=frozen()
 if a.smoke:
  jobs=[]
  for strategy in STRATEGIES:
   if any(x.get("kind")=="frozen_smoke" and x.get("strategy")==strategy and x.get("tag")=="frozen4" for x in old): continue
   env=f"LSS_VARIANT=mp2_r16_d4,LSS_SEED=123,LSS_STRATEGY={strategy},LSS_SMOKE=1,LSS_TAG=frozen4,LSS_RUNNER={snap/'run.py'},LSS_CODE_ROOT={ROOT}/notebooks/results/compact_lj_reconstruction/code_v2"
   try:
    job=qsub(["-v",env,str(HERE/"run.pbs")])
   except subprocess.CalledProcessError as exc:
    append({"kind":"frozen_smoke","status":"submission_failed","variant":"mp2_r16_d4","strategy":strategy,"seed":123,"submitted_at":now(),"error":str(exc),"runner_sha256":sha(snap/"run.py")})
    raise
   row={"kind":"frozen_smoke","status":"submitted","job_id":job,"tag":"frozen4","variant":"mp2_r16_d4","strategy":strategy,"seed":123,"submitted_at":now(),"runner_sha256":sha(snap/"run.py")};append(row);jobs.append(job)
  print(" ".join(jobs)); return
 # Hard gate: both smoke studies must have evaluated successfully and retained all contract rows.
 for strategy in STRATEGIES:
  d=OUT/"runs"/f"mp2_r16_d4_{strategy}_s123_smoke_frozen4"; c=d/"completed.json"; rows=d/"per_network_rows.csv"
  if not c.exists() or not rows.exists(): raise RuntimeError(f"Smoke has not completed: {strategy}")
  data=json.loads(c.read_text())
  if data.get("status")!="completed" or data.get("rows")!=800: raise RuntimeError(f"Smoke contract failed: {strategy}")
 for v in VARIANTS:
  for s in SEEDS:
   checkpoint=ae_path(v,s)
   if not checkpoint.exists() and v.endswith("d4"):
    raise RuntimeError(f"Missing frozen AE: {checkpoint}")
   for strategy in STRATEGIES:
    name=f"{v}_{strategy}_s{s}"; run=OUT/"runs"/name
    if run.exists() or any(x.get("kind")=="full" and x.get("name")==name for x in old): continue
    dep=[]
    if v.endswith("d2"):
     d2jobs={3456456:"4674650.zeus-master",123:"4674651.zeus-master",456:"4674652.zeus-master",786:"4674653.zeus-master",2026:"4674654.zeus-master"}
     dep=["-W",f"depend=afterok:{d2jobs[s]}"]
    env=f"LSS_VARIANT={v},LSS_SEED={s},LSS_STRATEGY={strategy},LSS_SMOKE=0,LSS_RUNNER={snap/'run.py'},LSS_CODE_ROOT={ROOT}/notebooks/results/compact_lj_reconstruction/code_v2"
    job=qsub([*dep,"-v",env,str(snap/"run.pbs")]);row={"kind":"full","status":"submitted","job_id":job,"name":name,"variant":v,"strategy":strategy,"seed":s,"depends_on":dep[-1] if dep else None,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint) if checkpoint.exists() else None,"submitted_at":now(),"runner_sha256":sha(snap/"run.py")};append(row);print(job,name,flush=True)
if __name__=="__main__":main()
