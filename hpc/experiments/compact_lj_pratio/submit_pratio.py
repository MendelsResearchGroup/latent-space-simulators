#!/usr/bin/env python3
"""Submit a p-ratio cell from an immutable evaluator snapshot."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RESULTS=ROOT/'notebooks/results/compact_lj_pratio'
RUNNER=ROOT/'hpc/experiments/compact_lj_pratio/run_pratio.py'
PBS=ROOT/'hpc/experiments/compact_lj_pratio/run_pratio.pbs'
PYTHON='/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--recipe',required=True);a=p.parse_args()
 snap=RESULTS/'snapshots'/('runner_'+sha(RUNNER)[:16]+'_v2');snap.mkdir(parents=True,exist_ok=True);frozen=snap/'run_pratio.py'
 if not frozen.exists():
  shutil.copy2(RUNNER,frozen)
  shutil.copytree(ROOT/'src',snap/'src')
 wrapper=snap/'run.pbs'
 if not wrapper.exists(): wrapper.write_text(PBS.read_text())
 out=RESULTS/'smoke'/a.name;out.mkdir(parents=True,exist_ok=True)
 env=','.join((f'LSS_PROJECT_ROOT={ROOT}',f'LSS_EVAL_CODE_ROOT={snap}',f'LSS_RUNNER={frozen}',f'LSS_PYTHON={PYTHON}',f'CHECKPOINT={Path(a.checkpoint).resolve()}',f'RECIPE={Path(a.recipe).resolve()}',f'OUT={out}'))
 job=subprocess.check_output(['qsub','-v',env,str(wrapper)],text=True).strip()
 (out/'submission.json').write_text(json.dumps({'job_id':job,'snapshot':str(frozen),'runner_sha256':sha(frozen),'checkpoint':str(Path(a.checkpoint).resolve()),'recipe':str(Path(a.recipe).resolve())},indent=2)+'\n');print(job)
if __name__=='__main__':main()
