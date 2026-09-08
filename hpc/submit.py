#!/usr/bin/env python3
"""Portable immutable-snapshot PBS planner/submitter for the research catalog.

It deliberately uses the locally available ``qsub`` executable.  ``--dry-run``
creates only a plan under the results root and never invokes PBS.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "experiments/catalog.json").read_text())
SEEDS = (3456456, 123, 456, 786, 2026)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p: Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def hashes(root: Path):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="hashes.json" and "__pycache__" not in p.parts and p.suffix!=".pyc"}
def append(path: Path,row: dict):
    with path.open("a") as f: f.write(json.dumps(row,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())
def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
def cell_id(arguments): return hashlib.sha256(json.dumps(arguments,sort_keys=True).encode()).hexdigest()[:16]

def freeze(project: Path, snapshot: Path, runner: str):
    if snapshot.exists():
        required=(snapshot/"src",snapshot/"runner"/runner,snapshot/"hpc_run_cell.py",snapshot/"collect.py",snapshot/"pbs"/"run_experiment.pbs",snapshot/"pbs"/"collect_experiment.pbs",snapshot/"hashes.json")
        if not all(x.exists() for x in required):
            raise RuntimeError("existing snapshot predates the portable schema; use a new code version, never overwrite it")
        recorded=json.loads((snapshot/"hashes.json").read_text())
        if recorded != hashes(snapshot): raise RuntimeError("immutable snapshot hashes differ; choose a new code version")
        return
    source=HERE/"experiments"/"runners"/runner
    if not source.is_file() or not (project/"src").is_dir(): raise RuntimeError("portable checkout must contain src/ and hpc/experiments/runners/")
    snapshot.mkdir(parents=True); shutil.copytree(project/"src",snapshot/"src",ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.egg-info")); (snapshot/"runner").mkdir()
    shutil.copy2(source,snapshot/"runner"/runner)
    helper=HERE/"experiments"/"runners"/"latent_diagnostic_metrics.py"
    if helper.exists(): shutil.copy2(helper,snapshot/"runner"/helper.name)
    shutil.copy2(HERE/"run_cell.py",snapshot/"hpc_run_cell.py")
    shutil.copy2(HERE/"collect.py",snapshot/"collect.py")
    shutil.copytree(HERE/"pbs", snapshot/"pbs")
    (snapshot/"hashes.json").write_text(json.dumps(hashes(snapshot),indent=2,sort_keys=True)+"\n")

def qsub(args):
    result=subprocess.run(["qsub",*args],text=True,capture_output=True,check=True)
    job=result.stdout.strip().splitlines()[-1]
    if not re.fullmatch(r"\d+\.[\w.-]+",job): raise RuntimeError("unexpected qsub output: "+result.stdout+result.stderr)
    return job
def expanded(study, smoke):
    base=CATALOG[study]["smoke" if smoke else "full"]
    if smoke:return base
    out=[]
    for args in base:
        for seed in SEEDS:
            x=list(args); x[x.index("--seed")+1]=str(seed); out.append(x)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("study",choices=sorted(CATALOG)); p.add_argument("--smoke",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--code-version",default="code_v1");p.add_argument("--project-root",default=os.environ.get("LSS_PROJECT_ROOT", str(HERE.parent)));p.add_argument("--results-root",default=os.environ.get("LSS_RESULTS_ROOT")); a=p.parse_args()
    project=Path(a.project_root).resolve(); results=Path(a.results_root).resolve() if a.results_root else project/"notebooks/results"; run_results=results/"hpc_runs"; base=run_results/a.study; snapshot=base/a.code_version; ledger=base/"jobs.jsonl"; split=Path(os.environ.get("LSS_SPLIT_MANIFEST", results/"lj_ae_08_bridge/split_manifest.json")).resolve(); base.mkdir(parents=True,exist_ok=True)
    with (base/"submit.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); freeze(project,snapshot,CATALOG[a.study]["runner"]); manifest=snapshot/"hashes.json"; cells=expanded(a.study,a.smoke)
        if not cells: raise RuntimeError(f"{a.study} has no supported smoke matrix; use its full recipe")
        python_executable=os.environ.get("LSS_PYTHON",sys.executable)
        recipe={"study":a.study,"kind":"smoke" if a.smoke else "full","code_version":a.code_version,"snapshot":str(snapshot),"snapshot_hashes":str(manifest),"snapshot_manifest_sha256":sha(manifest),"project_root":str(project),"results_root":str(run_results),"historical_results_root":str(results),"split_manifest":str(split),"python_executable":python_executable,"dependencies":CATALOG[a.study]["dependencies"],"cells":cells,"resources":{"queue":"mendels_q","placement":"free:shared","ncpus":8,"memory_gb":16,"walltime":"24:00:00"},"collector":{"afterany":not a.smoke,"ncpus":1,"memory_gb":2,"walltime":"00:15:00"},"created_at":now()}
        (base/("smoke_submission_recipe.json" if a.smoke else "submission_recipe.json")).write_text(json.dumps(recipe,indent=2,sort_keys=True)+"\n")
        (base/("smoke_intended_matrix.json" if a.smoke else "intended_matrix.json")).write_text(json.dumps(cells,indent=2)+"\n")
        prior=rows(ledger); submitted=[]
        for args in cells:
            ident=cell_id(args); existing=[r for r in prior if r.get("kind")=="run" and r.get("cell_id")==ident and r.get("code_version")==a.code_version and r.get("smoke")==a.smoke and r.get("job_id")]
            if existing: submitted.append(existing[-1]["job_id"]); continue
            cell={"study":a.study,"runner":CATALOG[a.study]["runner"],"arguments":args,"code_version":a.code_version,"project_root":str(project),"results_root":str(run_results),"historical_results_root":str(results),"split_manifest":str(split)}; cell_path=base/"cells"/(ident+".json");cell_path.parent.mkdir(exist_ok=True);cell_path.write_text(json.dumps(cell,indent=2)+"\n")
            row={"kind":"run","study":a.study,"cell_id":ident,"arguments":args,"smoke":a.smoke,"code_version":a.code_version,"submitted_at":now(),"queue":"mendels_q","placement":"free:shared","cpus":8,"memory_gb":16,"walltime":"24:00:00","snapshot":str(snapshot),"snapshot_manifest":str(manifest),"snapshot_manifest_sha256":sha(manifest),"cell_recipe":str(cell_path)}
            if a.dry_run: print(json.dumps({**row,"status":"planned"},sort_keys=True)); continue
            try: job=qsub(["-v",f"LSS_PROJECT_ROOT={project},LSS_RESULTS_ROOT={run_results},LSS_CODE_ROOT={snapshot},LSS_CELL_RECIPE={cell_path},LSS_PYTHON={python_executable}",str(snapshot/"pbs/run_experiment.pbs")])
            except Exception as e: append(ledger,{**row,"status":"qsub_failed","error":repr(e)});raise
            row.update(job_id=job,status="submitted");append(ledger,row);prior.append(row);submitted.append(job);print(json.dumps(row,sort_keys=True))
        if not a.smoke and not a.dry_run and submitted and not any(r.get("kind")=="collector" and r.get("code_version")==a.code_version and r.get("job_id") for r in prior):
            deps=list(dict.fromkeys(submitted)); row={"kind":"collector","study":a.study,"code_version":a.code_version,"dependencies":deps,"submitted_at":now(),"snapshot":str(snapshot),"snapshot_manifest_sha256":sha(manifest)}
            try: row["job_id"]=qsub(["-W","depend=afterany:"+":".join(deps),"-v",f"LSS_PROJECT_ROOT={project},LSS_RESULTS_ROOT={run_results},LSS_CODE_ROOT={snapshot},LSS_STUDY={a.study},LSS_CODE_VERSION={a.code_version},LSS_PYTHON={python_executable}",str(snapshot/"pbs/collect_experiment.pbs")]);row["status"]="submitted";append(ledger,row);print(json.dumps(row,sort_keys=True))
            except Exception as e:append(ledger,{**row,"status":"qsub_failed","error":repr(e)});raise
if __name__=="__main__": main()
