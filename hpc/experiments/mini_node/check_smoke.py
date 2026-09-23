"""Gate production on the complete two-trajectory validation smoke."""
from pathlib import Path
import json
import os
import pandas as pd
root=Path(os.environ['LSS_PROJECT_ROOT'])/'notebooks/results/mini_node/runs/d2_s123_smoke'
assert json.loads((root/'completed.json').read_text())['status']=='completed'
for strategy in ('one_step','multistep8'):
    assert json.loads((root/strategy/'completed.json').read_text())['ae_unchanged']
    table=pd.read_csv(root/strategy/'validation/per_step.csv')
    assert set(table.frame)==set(range(50))
    assert set(table['mode'])=={'ae_reconstruction','autonomous_delta_mlp'}
    assert set(table['split'])=={'validation'}
    assert len(table)==100
print('Smoke passed: both frozen-AE propagators and50-frame validation evaluation.')
