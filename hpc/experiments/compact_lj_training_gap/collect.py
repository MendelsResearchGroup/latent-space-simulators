#!/usr/bin/env python3
"""Aggregate frozen train diagnostics and the matched existing validation rows."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'notebooks/results/compact_lj_training_gap'
VAL = ROOT/'notebooks/results/compact_lj_pratio/runs'
FRAMES = (25, 50, 100, 150, 199)
SEEDS = (3456456, 123, 456, 786, 2026)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def completed(path):
    if not path.exists():
        raise RuntimeError(f'Missing completion marker: {path}')
    result = json.loads(path.read_text())
    if result.get('status') != 'completed':
        raise RuntimeError(f'Non-completed status in {path}: {result.get("status")!r}')
    return result

def select_and_validate(table, *, seed, split, expected_ids, checkpoint_hash, recipe_hash, label):
    required = {'seed', 'source', 'split', 'original_index', 'frame', 'target_index',
                'true_p_ratio', 'pred_p_ratio', 'coordinate_mse', 'checkpoint_sha256', 'recipe_sha256'}
    if missing := required - set(table.columns):
        raise RuntimeError(f'{label}: missing columns {sorted(missing)}')
    table = table[(table.seed == seed) & (table.source == 'lj_noisy') &
                  (table.split == split) & table.frame.isin(FRAMES)].copy()
    if not (table.target_index == table.frame).all():
        raise RuntimeError(f'{label}: target_index differs from frame')
    if table.duplicated(['original_index', 'frame']).any():
        raise RuntimeError(f'{label}: duplicate (original_index, frame) rows')
    if set(table.original_index) != set(expected_ids) or len(table) != len(expected_ids)*len(FRAMES):
        raise RuntimeError(f'{label}: IDs or row count differ from frozen split')
    if table.groupby('frame').original_index.nunique().to_dict() != {frame: len(expected_ids) for frame in FRAMES}:
        raise RuntimeError(f'{label}: incorrect per-frame ID counts')
    for column, expected in [('checkpoint_sha256', checkpoint_hash), ('recipe_sha256', recipe_hash)]:
        if set(table[column].dropna().astype(str)) != {expected}:
            raise RuntimeError(f'{label}: {column} differs from frozen training artifact')
    return table

def summarize(frame_rows):
    y, p = frame_rows.true_p_ratio.to_numpy(), frame_rows.pred_p_ratio.to_numpy()
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    denom = float(((y-y.mean())**2).sum()) if len(y) else float('nan')
    return {'total': len(frame_rows), 'valid': int(finite.sum()), 'nonfinite': int((~finite).sum()),
            'target_variance': float(np.var(y)) if len(y) else float('nan'),
            'r2_denominator': denom, 'p_ratio_r2': float(1-((p-y)**2).sum()/denom) if denom > 0 else float('nan'),
            'p_ratio_mae': float(np.abs(p-y).mean()) if len(y) else float('nan'),
            'coordinate_mse_mean': float(frame_rows.coordinate_mse.mean())}

def main():
    seed_rows, raw, provenance = [], [], []
    for seed in SEEDS:
        run = OUT/'runs'/f'mp2_r16_d4_s{seed}'
        val_run = VAL/f'compact_lj_reconstruction__mp2_r16_d4__s{seed}'
        train_file, train_eval_file = run/'per_network_rows.csv', run/'recipe.json'
        val_file, val_eval_file = val_run/'per_network_rows.csv', val_run/'recipe.json'
        for path in (train_file, train_eval_file, val_file, val_eval_file):
            if not path.exists(): raise RuntimeError(f'Missing required input: {path}')
        train_done, val_done = completed(run/'completed.json'), completed(val_run/'completed.json')
        train_eval, val_eval = json.loads(train_eval_file.read_text()), json.loads(val_eval_file.read_text())
        checkpoint, training_recipe_file = Path(train_eval['checkpoint']), Path(train_eval['training_recipe'])
        training_done = completed(checkpoint.parent/'completed.json')
        training_recipe = json.loads(training_recipe_file.read_text())
        checkpoint_hash, recipe_hash = sha(checkpoint), sha(training_recipe_file)
        if checkpoint_hash != training_done.get('ae_file_sha256'):
            raise RuntimeError(f'seed {seed}: checkpoint does not match training completion')
        for label, evaluation, marker in [('train', train_eval, train_done), ('validation', val_eval, val_done)]:
            if evaluation.get('checkpoint_sha256') != checkpoint_hash or marker.get('checkpoint_sha256') != checkpoint_hash:
                raise RuntimeError(f'seed {seed} {label}: checkpoint hash mismatch')
            if evaluation.get('training_recipe_sha256') != recipe_hash or marker.get('recipe_sha256') != recipe_hash:
                raise RuntimeError(f'seed {seed} {label}: recipe hash mismatch')
        specs = [x for x in training_recipe['source']['dataset_mixture'] if x['name'] == 'lj_noisy']
        if len(specs) != 1: raise RuntimeError(f'seed {seed}: expected one LJ source spec')
        train_ids, val_ids = list(specs[0]['split_indices']['train']), list(specs[0]['split_indices']['val'])
        if len(train_ids) != 60 or len(set(train_ids)) != 60 or len(val_ids) != 20 or len(set(val_ids)) != 20 or set(train_ids) & set(val_ids):
            raise RuntimeError(f'seed {seed}: expected disjoint 60-train/20-validation LJ IDs')
        train = select_and_validate(pd.read_csv(train_file), seed=seed, split='train', expected_ids=train_ids,
                                    checkpoint_hash=checkpoint_hash, recipe_hash=recipe_hash, label=f'seed {seed} train')
        val = select_and_validate(pd.read_csv(val_file), seed=seed, split='validation', expected_ids=val_ids,
                                  checkpoint_hash=checkpoint_hash, recipe_hash=recipe_hash, label=f'seed {seed} validation')
        raw.extend([train, val])
        provenance.append({'seed': seed, 'checkpoint_sha256': checkpoint_hash, 'training_recipe_sha256': recipe_hash,
                           'train_rows_sha256': sha(train_file), 'validation_rows_sha256': sha(val_file),
                           'train_evaluation_recipe_sha256': sha(train_eval_file), 'validation_evaluation_recipe_sha256': sha(val_eval_file),
                           'train_ids': train_ids, 'validation_ids': val_ids})
        for split, table in [('train', train), ('validation', val)]:
            for frame in FRAMES:
                seed_rows.append({'seed': seed, 'split': split, 'source': 'lj_noisy', 'frame': frame,
                                  **summarize(table[table.frame==frame])})
    seeds = pd.DataFrame(seed_rows)
    if len(seed_rows) != len(SEEDS)*2*len(FRAMES): raise RuntimeError('Missing seed/frame summaries')
    pd.concat(raw, ignore_index=True).to_csv(OUT/'per_network_rows_train_and_validation.csv', index=False)
    seeds.to_csv(OUT/'seed_results.csv', index=False)
    summary = seeds.groupby(['split','source','frame'], as_index=False).agg(
        seeds=('seed','nunique'), p_ratio_r2_mean=('p_ratio_r2','mean'), p_ratio_r2_std=('p_ratio_r2','std'),
        p_ratio_mae_mean=('p_ratio_mae','mean'), p_ratio_mae_std=('p_ratio_mae','std'),
        target_variance_mean=('target_variance','mean'), target_variance_std=('target_variance','std'),
        coordinate_mse_mean=('coordinate_mse_mean','mean'), valid_sum=('valid','sum'),
        nonfinite_sum=('nonfinite','sum'), total_sum=('total','sum'))
    if not (summary.valid_sum + summary.nonfinite_sum == summary.total_sum).all():
        raise RuntimeError('Aggregate finite/non-finite count mismatch')
    summary.to_csv(OUT/'source_frame_summary.csv', index=False)
    (OUT/'collection.json').write_text(json.dumps({'status': 'completed', 'completed_train_seeds': sorted(seeds.seed.unique().tolist()),
        'frames': FRAMES, 'method': 'physical endpoint sides quantile0.10; validation rows reused from compact_lj_pratio',
        'final_test_used': False, 'training_performed': False, 'selection_performed': False,
        'collector': str(Path(__file__).resolve()), 'collector_sha256': sha(Path(__file__)),
        'input_provenance': provenance}, indent=2)+'\n')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
