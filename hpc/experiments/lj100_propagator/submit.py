#!/usr/bin/env python3
import argparse,hashlib,json,re,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'notebooks/results/lj100_propagator'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def freeze():
 d=OUT/'code_v6';files=('run.py','evaluate.py','run.pbs','submit.py')
 if d.exists():
  m=json.loads((d/'manifest.json').read_text());assert {k:sha(d/k) for k in m}==m;return d
 d.mkdir(parents=True)
 for n in files:shutil.copy2(HERE/n,d/n)
 shutil.copy2(ROOT/'tools/analysis/collect_lj100_propagator.py',d/'collect.py');shutil.copy2(OUT/'protocol.json',d/'protocol.json');shutil.copytree(ROOT/'src/lss',d/'src/lss',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 m={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file() and p.name!='manifest.json'};(d/'manifest.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n');return d
def q(args):
 x=subprocess.check_output(['qsub',*args],text=True).strip().splitlines()[-1]
 if not re.fullmatch(r'\d+\.[\w.-]+',x):raise RuntimeError(x)
 return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--production',action='store_true');a=p.parse_args()
 if a.smoke==a.production:raise SystemExit('choose one')
 d=freeze();ledger=OUT/'jobs.jsonl';old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
 if a.production:raise SystemExit('production is root smoke-review gated')
 for obj in ('latent_state','decoded_positions'):
  if any(x.get('kind')=='smoke_v6' and x.get('objective')==obj for x in old):continue
  job=q(['-v',f'LSS_SEED=123,LSS_OBJECTIVE={obj},LSS_LR=0.0001,LSS_SMOKE=1,LSS_RUNNER={d/"run.py"},LSS_CODE_ROOT={d}',str(d/'run.pbs')]);row={'kind':'smoke_v6','objective':obj,'job_id':job,'snapshot':str(d),'submitted_at':datetime.now(timezone.utc).isoformat()}
  with ledger.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
  print(job,obj)
if __name__=='__main__':main()
