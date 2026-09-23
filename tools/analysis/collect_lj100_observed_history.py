"""Collect the predeclared broad LJ100 matrix without ranking or selecting runs."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd


def summarize(group):
    true = group['true_p_ratio'].to_numpy(float)
    pred = group['pred_p_ratio'].to_numpy(float)
    valid = np.isfinite(true) & np.isfinite(pred)
    y, estimate = true[valid], pred[valid]
    variance = np.square(y - y.mean()).sum() if len(y) else 0.0
    response = 1 - np.square(y - estimate).sum() / variance if variance > 0 else np.nan
    count = group['displacement_count'].sum()
    sst = sum(group[f'displacement_square_sum_{axis}'].sum()
              - group[f'displacement_sum_{axis}'].sum() ** 2 / count
              for axis in ('x', 'y'))
    displacement = 1 - group['displacement_sse'].sum() / sst if sst > 0 else np.nan
    return dict(p_ratio_r2=response, p_ratio_valid=int(valid.sum()), p_ratio_total=len(group),
                displacement_r2=displacement)


def collect(root):
    study = root / 'notebooks/results/lj100_observed_history'
    protocol = json.loads((study / 'protocol.json').read_text())
    ids = {spec['name']: set(spec['split_indices']['val'])
           for spec in protocol['source']['dataset_mixture']}
    status, metrics = [], []
    for arm in protocol['matrix']:
        folder = study / 'runs' / arm['run_name']
        if not (folder / 'completed.json').exists():
            outcome = 'failed' if (folder / 'failed.json').exists() else 'unfinished'
            status.append({**arm, 'status': outcome})
            continue
        completed = json.loads((folder / 'completed.json').read_text())
        table = pd.read_csv(folder / 'per_network_rows.csv')
        assert len(table) == 60*98*(3 if arm['observed_frames']==3 else 2), (folder, len(table))
        assert not table.duplicated(['source', 'original_index', 'frame', 'kind']).any()
        assert set(table['seed']) == {arm['seed']}
        assert set(table['family']) == {arm['family']}
        assert set(table['observed_frames']) == {arm['observed_frames']}
        assert set(table['split']) == {'validation'}
        assert set(table['source']) == set(ids)
        assert set(table['frame']) == set(range(3, 101))
        expected_kinds = {'reconstruction','prefix_prediction','linear_history'} if arm['observed_frames']==3 else {'reconstruction','prefix_prediction'}
        assert set(table['kind']) == expected_kinds
        for (source, frame, kind), group in table.groupby(['source', 'frame', 'kind']):
            assert set(group['original_index']) == ids[source]
            metrics.append({**arm, 'source': source, 'split': 'validation',
                            'frame': frame, 'kind': kind, **summarize(group)})
        status.append({**arm, 'status': completed['status'], 'job_id': completed['job_id']})
    pd.DataFrame(status).to_csv(study / 'collection_status.csv', index=False)
    if not metrics:
        raise RuntimeError('No completed production runs; smoke outputs are not study results.')
    per_seed = pd.DataFrame(metrics)
    per_seed.to_csv(study / 'per_seed_steps.csv', index=False)
    aggregate = (per_seed.groupby(['family', 'observed_frames', 'source', 'split', 'frame', 'kind'])
                 .agg(p_ratio_r2_mean=('p_ratio_r2', 'mean'),
                      p_ratio_r2_std=('p_ratio_r2', 'std'),
                      finite_r2_seeds=('p_ratio_r2', 'count'),
                      completed_seeds=('seed', 'nunique'),
                      p_ratio_valid=('p_ratio_valid', 'sum'),
                      p_ratio_total=('p_ratio_total', 'sum'),
                      displacement_r2_mean=('displacement_r2', 'mean'),
                      displacement_r2_std=('displacement_r2', 'std')).reset_index())
    aggregate.to_csv(study / 'seed_summary.csv', index=False)
    horizon = aggregate[aggregate['frame'].eq(100)]
    lines = ['# Shared propagator through step100', '',
             'Validation development; final test untouched. All arms retained.',
             'p-ratio R² is physical endpoint response, original side groups at quantile0.10.',
             'Counts sum repeated evaluations across model seeds;20 unique networks/source.', '',
             '| Family | Observed frames | Source | Mode | p-ratio R² mean ± SD | Seeds | Valid/total |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in horizon.itertuples():
        lines.append(f'| {row.family} | {row.observed_frames} | {row.source} | {row.kind} | '
                     f'{row.p_ratio_r2_mean:.3f} ± {row.p_ratio_r2_std:.3f} | '
                     f'{row.finite_r2_seeds}/{row.completed_seeds} | '
                     f'{row.p_ratio_valid}/{row.p_ratio_total} |')
    lines += ['', '## Run outcomes', '', pd.DataFrame(status).to_string(index=False), '']
    (study / 'review.md').write_text('\n'.join(lines))
    (study / 'collection.json').write_text(json.dumps({
        'complete': all(row['status'] == 'completed' for row in status),
        'expected_runs': len(status), 'completed_runs': sum(row['status'] == 'completed' for row in status),
        'protocol_sha256': hashlib.sha256((study / 'protocol.json').read_bytes()).hexdigest(),
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    collect(parser.parse_args().root)
