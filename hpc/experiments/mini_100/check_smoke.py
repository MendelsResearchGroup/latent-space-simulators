import argparse,json,os
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',required=True);a=p.parse_args()
out=Path(os.environ['LSS_PROJECT_ROOT'])/'notebooks/results/mini_100'/f'{a.study}_{a.mode}_smoke'
assert json.loads((out/'pipeline_completed.json').read_text())['status']=='completed'
for strategy in ['one_step']:
    assert json.loads((out/strategy/'completed.json').read_text())['ae_unchanged']
    table=pd.read_csv(out/strategy/'validation/per_step.csv')
    assert set(table.frame)==set(range(100)) and len(table)==200
    assert set(table['mode'])=={'ae_reconstruction','autonomous_delta_mlp'}
print('AE and one-step propagator smoke checks passed through99; production starts.',flush=True)
