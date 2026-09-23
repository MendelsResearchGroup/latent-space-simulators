"""Prepare exact temporal prefixes of120 distinct node-optimized networks."""
import hashlib
import json
import os
from pathlib import Path
import torch

ROOT=Path(os.environ['LSS_PROJECT_ROOT'])
STUDY=ROOT/'notebooks/results/mini_node'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

full_path=ROOT/'data/mini_data_full200.manifest.json'
prefix_path=ROOT/'data/mini_data_converted.manifest.json'
full=json.loads(full_path.read_text());prefix=json.loads(prefix_path.read_text())
assert full['prefix_manifest_sha256']==sha(prefix_path)
records=prefix['records']['node_optimized']
lookup={r['file_path'].removeprefix('./'):r for r in records}
selected=[lookup[path] for path in full['selected_paths']['node_optimized'][:120]]
assert len(selected)==120 and len({r['topology_sha256'] for r in selected})==120
input_path=Path(prefix['outputs']['node_optimized']['path'])
assert sha(input_path)==prefix['outputs']['node_optimized']['sha256']
all_data=torch.load(input_path,map_location='cpu',weights_only=False)
cohort=[all_data[r['trajectory_index']] for r in selected]
assert all(len(sim)==50 for sim in cohort)
for sim,record in zip(cohort,selected):
    assert sim[0].source_chunk==Path(record['file_path']).name
    assert [g.source_frame_index for g in sim]==list(range(50))
p20=ROOT/'data/mini_node_cohort120_first20.pt'
p50=ROOT/'data/mini_node_cohort120_first50.pt'
torch.save([sim[:20] for sim in cohort],p20);torch.save(cohort,p50)
source=dict(dataset_name='mini_node',source_name='mini_node',label='Node-optimized mini networks',
    coordinate_normalization='position_normalization',path=str(p20),dataset_mixture=[dict(
    name='mini_node',label='Node-optimized mini networks',path=str(p20),train_count=20,val_count=20,
    split_indices=dict(train=list(range(20)),val=list(range(20,40)),test=[]))])
prepared=dict(source=source,prefix_sha256=sha(p20),evaluation_path=str(p50),evaluation_sha256=sha(p50),
    test_indices=list(range(40,120)),selected_records=selected,full_manifest_sha256=sha(full_path),
    input_manifest_sha256=sha(prefix_path),input_prefix_sha256=sha(input_path),job_id=os.environ['PBS_JOBID'])
(STUDY/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print('Prepared120 distinct node networks:20train20validation80test; exact first20/50frames.',flush=True)
