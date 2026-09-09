#!/usr/bin/env python3
"""Collect frozen AE endpoint p-ratio evaluations without training or selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'notebooks/results'
OUTPUT = RESULTS / 'compact_lj_pratio'
STUDIES = {'compact_lj_reconstruction': 'code_v2', 'compact_lj_spatial_decoder': 'code_v1'}
FRAMES = (25, 50, 100, 150, 199)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(group, true_col, pred_col):
    y = pd.to_numeric(group[true_col], errors='coerce').to_numpy(float)
    pred = pd.to_numeric(group[pred_col], errors='coerce').to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(pred)
    yf, pf = y[valid], pred[valid]
    denom = float(np.square(yf - yf.mean()).sum()) if len(yf) else 0.0
    return {
        'total': len(group), 'valid': int(valid.sum()),
        'true_nonfinite': int((~np.isfinite(y)).sum()),
        'pred_nonfinite': int((~np.isfinite(pred)).sum()),
        'p_ratio_r2': float(1 - np.square(yf-pf).sum()/denom) if len(yf) >= 2 and denom > 0 else np.nan,
        'p_ratio_mae': float(np.abs(yf-pf).mean()) if len(yf) else np.nan,
        'target_variance': float(np.var(yf)) if len(yf) else np.nan,
        'r2_denominator': denom,
    }


def main():
    collection = OUTPUT / 'collection'
    collection.mkdir(parents=True, exist_ok=True)
    raw_file = OUTPUT/'raw_target_audit/raw_target_rows.csv'
    raw_targets = pd.read_csv(raw_file)
    statuses, frames, input_hashes = [], [], {}
    for study, version in STUDIES.items():
        for recipe in sorted((RESULTS/study).glob(f'*_s*_{version}/recipe.json')):
            r = json.loads(recipe.read_text())
            if r.get('smoke'):
                continue
            run = OUTPUT/'runs'/f'{study}__{r["variant"]}__s{r["seed"]}'
            status = {'study': study, 'variant': r['variant'], 'seed': r['seed'],
                      'training_job': r['job_id'], 'run_dir': str(run.relative_to(ROOT))}
            if not (run/'completed.json').exists():
                status['status'] = 'failed' if (run/'failed.json').exists() else 'incomplete'
                statuses.append(status)
                continue
            try:
                done = json.loads((run/'completed.json').read_text())
                if done.get('status') != 'completed':
                    raise ValueError('Non-completed terminal marker')
                rows_file = run/'per_network_rows.csv'
                rows = pd.read_csv(rows_file)
                rows['study'], rows['variant'], rows['seed'] = study, r['variant'], r['seed']
                rows['split'] = rows['split'].replace({'val': 'validation'})
                expected = {(s['name'], 'validation'): set(s['split_indices']['val'])
                            for s in r['source']['dataset_mixture']}
                if r.get('mixed_evaluation'):
                    expected.update({(s['name'], 'mixed_transfer'): set(s['split_indices']['val'])
                                     for s in r['mixed_evaluation']['dataset_mixture']})
                if set(map(tuple, rows[['source', 'split']].drop_duplicates().to_numpy())) != set(expected):
                    raise ValueError('Wrong source or evaluation-split coverage')
                if rows.duplicated(['source', 'split', 'original_index', 'frame']).any():
                    raise ValueError('Duplicate network/frame evaluations')
                for (source, split), identities in expected.items():
                    group = rows[(rows.source == source) & (rows.split == split)]
                    if set(group.frame) != set(FRAMES):
                        raise ValueError('Missing or unexpected horizons')
                    for frame, g in group.groupby('frame'):
                        if set(g.original_index) != identities or len(g) != len(identities):
                            raise ValueError(f'Wrong identities: {source}/{frame}')
                original = pd.read_csv(recipe.parent/'source_frame_rows.csv')
                original = original[original.split.isin(['validation', 'mixed_transfer']) & original.frame.isin(FRAMES)]
                parity = rows.merge(original, on=['source', 'split', 'original_index', 'frame'],
                                    suffixes=('_new', '_original'), validate='one_to_one')
                if len(parity) != len(original) or not {25, 50, 100, 199}.issubset(set(parity.frame)):
                    raise ValueError('Saved coordinate rows do not cover the four common horizons')
                max_delta = float((parity.coordinate_mse_new-parity.coordinate_mse_original).abs().max())
                if not np.isfinite(max_delta) or max_delta > 1e-10:
                    raise ValueError(f'Frozen-coordinate MSE parity failed: max absolute difference {max_delta}')
                status['max_coordinate_mse_delta'] = max_delta
                checkpoint_hash = json.loads((recipe.parent/'completed.json').read_text())['ae_file_sha256']
                if set(rows.checkpoint_sha256) != {checkpoint_hash}:
                    raise ValueError('Evaluation checkpoint hash differs from original training artifact')
                raw_check = rows.merge(raw_targets, on=['source', 'original_index', 'frame'], validate='one_to_one')
                if len(raw_check) != len(rows):
                    raise ValueError('Independent raw-data audit does not cover all rows')
                max_p_delta = float((raw_check.true_p_ratio-raw_check.raw_physical_true_p_ratio).abs().max())
                if not np.isfinite(max_p_delta) or max_p_delta > 1e-4:
                    raise ValueError(f'Raw-data p-ratio parity failed: max difference {max_p_delta}')
                status['max_raw_true_pratio_delta'] = max_p_delta
                frames.append(rows)
                input_hashes[str(rows_file.relative_to(ROOT))] = digest(rows_file)
                status['status'] = 'completed'
                status['rows'] = len(rows)
                status['evaluation_job'] = done.get('job_id', done.get('pbs_job_id'))
            except Exception as exc:
                status['status'] = 'invalid'
                status['error'] = f'{type(exc).__name__}: {exc}'
            statuses.append(status)
    status_table = pd.DataFrame(statuses)
    assert len(status_table) == 70, f'Expected70 known checkpoints, found{len(status_table)}'
    status_table.to_csv(collection/'run_status.csv', index=False)
    keys = ['study', 'variant', 'seed', 'source', 'split', 'frame']
    if frames:
        rows = pd.concat(frames, ignore_index=True)
        rows['transfer_only'] = rows.split.eq('mixed_transfer')
        rows.to_csv(collection/'per_network_rows.csv', index=False)
        records = []
        for key, group in rows.groupby(keys):
            record = dict(zip(keys, key))
            record['transfer_only'] = record['split'] == 'mixed_transfer'
            record.update(scores(group, 'true_p_ratio', 'pred_p_ratio'))
            secondary = scores(group, 'model_true_p_ratio', 'model_pred_p_ratio')
            record.update({f'model_{name}': value for name, value in secondary.items()})
            record['coordinate_mse_mean'] = float(group.coordinate_mse.mean())
            for col in ('near_zero_strain_true', 'near_zero_strain_pred'):
                if col in group:
                    record[col] = int(group[col].fillna(False).astype(bool).sum())
            records.append(record)
        seeds = pd.DataFrame(records)
        seeds.to_csv(collection/'source_frame_seed_results.csv', index=False)
        aggs = {'seeds': ('seed', 'nunique'), 'valid_sum': ('valid', 'sum'),
                'valid_min': ('valid', 'min'), 'total_sum': ('total', 'sum'), 'total_min': ('total', 'min'),
                'r2_valid_seeds': ('p_ratio_r2', 'count'),
                'true_nonfinite_sum': ('true_nonfinite', 'sum'), 'pred_nonfinite_sum': ('pred_nonfinite', 'sum')}
        for col in ('p_ratio_r2', 'p_ratio_mae', 'target_variance', 'coordinate_mse_mean',
                    'model_p_ratio_r2', 'model_p_ratio_mae'):
            aggs[f'{col}_mean'] = (col, 'mean')
            aggs[f'{col}_std'] = (col, 'std')
        for col in ('near_zero_strain_true', 'near_zero_strain_pred'):
            if col in seeds:
                aggs[f'{col}_sum'] = (col, 'sum')
        aggregate = seeds.groupby(['study', 'variant', 'source', 'split', 'transfer_only', 'frame']).agg(**aggs).reset_index()
        aggregate.to_csv(collection/'source_frame_aggregate.csv', index=False)
    counts = status_table.status.value_counts().to_dict()
    metadata = {'expected': 70, 'status_counts': counts,
                'complete': counts.get('completed', 0) == 70,
                'primary_metric': 'physical-coordinate endpoint directional-side p-ratio R2',
                'secondary_metric': 'historical model-coordinate endpoint p-ratio R2',
                'selection': 'none; all frozen state-selected checkpoints retained',
                'final_test_used': False, 'frames': list(FRAMES),
                'count_qualification': 'Five model seeds repeat the same20 validation networks/source;100 evaluations are not100 unique networks.',
                'collector_sha256': digest(Path(__file__)), 'raw_target_audit_sha256': digest(raw_file),
                'input_sha256': input_hashes}
    (collection/'collection.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'expected': 70, 'status_counts': counts}, indent=2))


if __name__ == '__main__':
    main()
