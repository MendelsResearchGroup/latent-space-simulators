#!/usr/bin/env python3
"""Freeze tiny-edge pilot code, then submit its three declared seeds."""
import hashlib, json, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('/rg/mendels_prj/alexander.z/latent-space-simulators'); HERE=ROOT/'hpc/experiments/tiny_edge_mlp'; OUT=ROOT/'notebooks/results/tiny_edge_mlp'; CODE=OUT/'code_v1'
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''): h.update(x)
 return h.hexdigest()
if CODE.exists(): raise FileExistsError(CODE)
CODE.mkdir(parents=True)
for p in (HERE/'run.py', HERE/'pipeline.pbs', ROOT/'src/lss/dynamics/tiny_edge_ae.py', ROOT/'notebooks/results/all4_coverage_1000/code_v5/evaluate.py', ROOT/'notebooks/results/all4_coverage_1000/code_v5/data_manifest.json'):
 shutil.copy2(p, CODE/p.name)
manifest={p.name:sha(p) for p in CODE.iterdir() if p.is_file()}; (CODE/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
ledger={'study':'tiny_edge_mlp','status':'submitting','submitted_at':datetime.now(timezone.utc).isoformat(),'code_root':str(CODE.relative_to(ROOT)),'code_manifest_sha256':sha(CODE/'manifest.json'),'seeds':[123,456,786],'resources':{'queue':'mendels_q','ncpus':8,'mem':'16gb','place':'free:shared'},'jobs':{}}
for seed in ledger['seeds']:
 job=subprocess.check_output(['/usr/local/bin/qsub','-v',f'LSS_CODE_ROOT={CODE},LSS_SEED={seed}',str(CODE/'pipeline.pbs')],text=True).strip(); ledger['jobs'][str(seed)]=job
 (OUT/'job_ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
ledger['status']='submitted'; (OUT/'job_ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
print(json.dumps(ledger,indent=2,sort_keys=True))
