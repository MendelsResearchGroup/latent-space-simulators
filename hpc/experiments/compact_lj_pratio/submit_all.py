#!/usr/bin/env python3
"""Ledgered immutable submission of every frozen compact-LJ AE checkpoint."""
from __future__ import annotations
import fcntl, hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; RESULTS=ROOT/'notebooks/results/compact_lj_pratio'
RUNNER=ROOT/'hpc/experiments/compact_lj_pratio/run_pratio.py'; PBS=ROOT/'hpc/experiments/compact_lj_pratio/run_pratio.pbs'; PYTHON='/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python'
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 recipes=[]
 for study,version in [('compact_lj_reconstruction','code_v2'),('compact_lj_spatial_decoder','code_v1')]:
  for recipe in sorted((ROOT/'notebooks/results'/study).glob(f'*_{version}/recipe.json')):
   if '_smoke' not in str(recipe): recipes.append((study,recipe,recipe.parent/'ae.pt'))
 if len(recipes)!=70: raise RuntimeError(f'expected 70 full recipes, found {len(recipes)}')
 snap=RESULTS/'snapshots'/('runner_'+sha(RUNNER)[:16]+'_broad'); snap.mkdir(parents=True,exist_ok=True)
 frozen=snap/'run_pratio.py'
 if not frozen.exists():
  shutil.copy2(RUNNER,frozen);shutil.copytree(ROOT/'src',snap/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  hashes={str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file()}
  (snap/'hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
 wrapper=snap/'run.pbs'
 if not wrapper.exists(): shutil.copy2(PBS,wrapper)
 hashes={str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file() and p.name!='hashes.json'}
 (snap/'hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
 ledger=RESULTS/'jobs.jsonl'; RESULTS.mkdir(parents=True,exist_ok=True)
 (RESULTS/'intended_matrix.json').write_text(json.dumps([{'study':s,'recipe':str(r),'checkpoint':str(c)} for s,r,c in recipes],indent=2)+'\n')
 with ledger.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.seek(0); prior={json.loads(x)['name'] for x in f if x.strip() and json.loads(x).get('status')=='submitted'}
  for study,recipe,ck in recipes:
   meta=json.loads(recipe.read_text());name=f"{study}__{meta['variant']}__s{meta['seed']}";out=RESULTS/'runs'/name
   if name in prior: continue
   out.mkdir(parents=True,exist_ok=True)
   if not ck.is_file(): raise RuntimeError(f'missing checkpoint: {ck}')
   completed=ck.parent/'completed.json'
   if not completed.is_file() or json.loads(completed.read_text()).get('status')!='completed': raise RuntimeError(f'checkpoint not completed: {ck}')
   if sha(ck)!=json.loads(completed.read_text())['ae_file_sha256']: raise RuntimeError(f'checkpoint hash mismatch: {ck}')
   env=','.join((f'LSS_PROJECT_ROOT={ROOT}',f'LSS_EVAL_CODE_ROOT={snap}',f'LSS_RUNNER={frozen}',f'LSS_PYTHON={PYTHON}',f'CHECKPOINT={ck}',f'RECIPE={recipe}',f'OUT={out}'))
   (out/'submission_recipe.json').write_text(json.dumps({'name':name,'study':study,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'recipe':str(recipe),'recipe_sha256':sha(recipe),'snapshot':str(snap),'runner_sha256':sha(frozen),'resources':{'queue':'mendels_q','ncpus':1,'memory_gb':16,'place':'free:shared'}},indent=2,sort_keys=True)+'\n')
   job=subprocess.check_output(['qsub','-v',env,str(wrapper)],text=True).strip()
   f.write(json.dumps({'status':'submitted','name':name,'study':study,'variant':meta['variant'],'seed':meta['seed'],'job_id':job,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'recipe':str(recipe),'recipe_sha256':sha(recipe),'snapshot':str(snap),'runner_sha256':sha(frozen),'submitted_at':datetime.now(timezone.utc).isoformat()},sort_keys=True)+'\n');f.flush()
   print(job,name,flush=True)
if __name__=='__main__': main()
