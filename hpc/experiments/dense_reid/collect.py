#!/usr/bin/env python3
"""Collect every predeclared dense-Reid cell without selecting variants."""
from pathlib import Path
import json
import os
import pandas as pd

ROOT=Path(os.environ['LSS_PROJECT_ROOT'])
OUT=ROOT/'notebooks/results/dense_reid'
rows=[]
status=[]
for dim in (2,4):
    for seed in (123,456,786):
        base=OUT/'runs'/f'd{dim}_s{seed}'
        completed=base/'completed.json'
        status.append({'dimension':dim,'seed':seed,'completed':completed.is_file()})
        if not completed.is_file():
            continue
        for strategy in ('one_step','multistep8'):
            for split in ('validation','test'):
                table=pd.read_csv(base/strategy/split/'per_step.csv')
                assert set(table['frame'])==set(range(101))
                assert set(table['mode'])=={'ae_reconstruction','autonomous_delta_mlp'}
                assert set(table['split'])=={split}
                assert set(table['networks'])=={10 if split=='validation' else 20}
                if strategy=='multistep8':
                    table=table[table['mode']=='autonomous_delta_mlp']
                table['strategy']=[('AE' if mode=='ae_reconstruction' else strategy) for mode in table['mode']]
                rows.append(table.assign(latent_dim=dim,seed=seed))
(OUT/'collection_status.json').write_text(json.dumps({'cells':status,'complete':all(r['completed'] for r in status)},indent=2)+'\n')
if rows:
    all_scores=pd.concat(rows,ignore_index=True)
    all_scores.to_csv(OUT/'per_seed_steps.csv',index=False)
    metrics=['displacement_pooled_r2','increment_pooled_r2','p_ratio_r2']
    summary=all_scores.groupby(['latent_dim','strategy','split','source','frame'])[metrics].agg(['mean','std','count'])
    summary.columns=['_'.join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(OUT/'seed_summary.csv',index=False)
    print(summary.xs(100,level='frame').to_string())
print(json.dumps({'complete':all(r['completed'] for r in status),'completed_cells':sum(r['completed'] for r in status),'total_cells':6}))
