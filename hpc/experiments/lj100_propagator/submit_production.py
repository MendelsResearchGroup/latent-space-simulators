#!/usr/bin/env python3
"""Root-review-gated submission of the frozen v6 matrix and collector."""
import hashlib,json,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'notebooks/results/lj100_propagator'; SNAP=OUT/'code_v6'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def q(args):
 x=subprocess.check_output(['qsub',*args],text=True).strip().splitlines()[-1]
 if not re.fullmatch(r'\d+\.[\w.-]+',x):raise RuntimeError(f'bad PBS id: {x}')
 return x
def main():
 manifest=json.loads((SNAP/'manifest.json').read_text())
 if {k:sha(SNAP/k) for k in manifest}!=manifest:raise RuntimeError('v6 hash validation failed')
 if sha(OUT/'protocol.json')!=sha(SNAP/'protocol.json'):raise RuntimeError('root protocol differs from frozen v6 protocol')
 smoke={}
 for obj in ('latent_state','decoded_positions'):
  run=OUT/'runs'/f'mp2d_s123_{obj}_lr0.0001_smoke_code_v6'; done=json.loads((run/'completed.json').read_text()); recipe=json.loads((run/'recipe.json').read_text())
  if done['status']!='completed' or recipe['code_manifest_sha256']!=sha(SNAP/'manifest.json'):raise RuntimeError('smoke gate failed')
  smoke[obj]={'job_id':done['job_id'],'completed':done,'baseline_parity':recipe['baseline_parity'],'frozen_invariant':recipe['frozen_before']==recipe['frozen_after'],'pure_position_gradient':recipe['pure_position_gradient']}
 if not smoke['decoded_positions']['pure_position_gradient'] or not all(x['frozen_invariant'] for x in smoke.values()):raise RuntimeError('smoke invariant/gradient gate failed')
 (OUT/'smoke_gate.json').write_text(json.dumps({'status':'approved','protocol_sha256':sha(OUT/'protocol.json'),'snapshot_manifest_sha256':sha(SNAP/'manifest.json'),'submitter_sha256':sha(Path(__file__)),'smokes':smoke},indent=2)+'\n')
 ledger=OUT/'jobs.jsonl';old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else [];jobs=[]
 for arm in json.loads((OUT/'protocol.json').read_text())['matrix']:
  if any(x.get('kind')=='production_v6' and x.get('run_name')==arm['run_name'] for x in old):continue
  env=f'LSS_SEED={arm["seed"]},LSS_OBJECTIVE={arm["objective"]},LSS_LR={arm["lr"]},LSS_SMOKE=0,LSS_RUNNER={SNAP/"run.py"},LSS_CODE_ROOT={SNAP}'
  job=q(['-v',env,str(SNAP/'run.pbs')]);row={'kind':'production_v6','run_name':arm['run_name'],'seed':arm['seed'],'objective':arm['objective'],'lr':arm['lr'],'job_id':job,'snapshot':str(SNAP),'submitter_sha256':sha(Path(__file__)),'submitted_at':datetime.now(timezone.utc).isoformat()}
  with ledger.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
  jobs.append(job);print(job,arm['run_name'])
 all_jobs=[x['job_id'] for x in old if x.get('kind')=='production_v6']+jobs
 if all_jobs and not any(x.get('kind')=='collector_v6' for x in old):
  c=OUT/'collector_v6'
  if not c.exists():
   c.mkdir();shutil.copy2(HERE/'collect.pbs',c/'collect.pbs');shutil.copy2(SNAP/'collect.py',c/'collect.py');(c/'manifest.json').write_text(json.dumps({x:sha(c/x) for x in ('collect.pbs','collect.py')},indent=2)+'\n')
  m=json.loads((c/'manifest.json').read_text());assert {x:sha(c/x) for x in m}==m
  job=q(['-W','depend=afterany:'+':'.join(all_jobs),'-v',f'LSS_COLLECTOR={c/"collect.py"}',str(c/'collect.pbs')]);
  with ledger.open('a') as f:f.write(json.dumps({'kind':'collector_v6','job_id':job,'depends_on':all_jobs,'snapshot':str(c),'submitter_sha256':sha(Path(__file__)),'submitted_at':datetime.now(timezone.utc).isoformat()},sort_keys=True)+'\n')
  print(job,'collector')
if __name__=='__main__':main()
