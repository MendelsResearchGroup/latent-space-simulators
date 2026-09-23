#!/usr/bin/env python3
"""Correlate backward latent differences with ending-frame physical p-ratio."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'notebooks/results/mp2_latent_analysis'
frames = pd.read_csv(BASE/'frame_rows.csv')
frames = frames[frames.source.isin(['reid', 'depablo_low_temp', 'lj_noisy'])].sort_values(['source', 'simulation_id', 'frame']).copy()
groups = frames.groupby(['source', 'simulation_id'])
dt = groups.frame.diff()
assert dt.dropna().eq(1).all()
for latent in ['z0', 'z1']:
    frames['d'+latent] = groups[latent].diff()/dt
frames = frames[frames.frame.between(5, 199)]
rows = []
for (source, frame), group in frames.groupby(['source', 'frame']):
    for latent in ['z0', 'z1']:
        valid = np.isfinite(group['d'+latent]) & np.isfinite(group.p_ratio)
        rows.append(dict(source=source, frame=frame, latent=latent,
                         r=group.loc[valid, 'd'+latent].corr(group.loc[valid, 'p_ratio']),
                         valid=int(valid.sum()), total=len(group)))
table = pd.DataFrame(rows)
assert len(table)==1170 and table.valid.eq(20).all() and table.total.eq(20).all()
out = BASE/'correlation_search'
table.to_csv(out/'frame_rate_correlations.csv', index=False)
metadata = dict(model_seed=786, frames=[5,199], slope='(z[t]-z[t-1])/(step[t]-step[t-1]); compute before filtering to preserve frame4→5',
                target='p_ratio[t], established physical endpoint side estimator relative to frame0',
                smoothing=False, correlation='Pearson across20networks within each source and step; median across195steps in plot',
                frame_network_pairs=len(frames), coefficients=len(table), valid_total_per_coefficient=[20,20],
                inference='descriptive; repeated frames are not independent observations; no pooled significance claim',
                input_sha256=hashlib.sha256((BASE/'frame_rows.csv').read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                training=False, new_model_inference=False, final_test_used=False, notebook_executed=False)
(out/'frame_rate_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved 1170 within-step correlations; all20/20valid. Notebook not executed.')
