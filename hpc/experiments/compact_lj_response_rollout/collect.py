#!/usr/bin/env python3
"""Collect audited reconstruction/rollout curves without selecting checkpoints."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get('LSS_PROJECT_ROOT', Path(__file__).resolve().parents[3]))
RESULTS = ROOT / 'notebooks/results'
OUT = RESULTS / 'compact_lj_response_rollout'
SEEDS = {3456456, 123, 456, 786, 2026}
FRAMES = {25, 50, 100, 150, 199}
SOURCES = {'reid', 'depablo_low_temp', 'lj_noisy', 'depablo_mixed_temp'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(g):
    y = g.true_p_ratio.to_numpy(float)
    p = g.pred_p_ratio.to_numpy(float)
    mask = np.isfinite(y) & np.isfinite(p)
    yf, pf = y[mask], p[mask]
    denominator = np.square(yf-yf.mean()).sum() if len(yf) else 0
    return dict(p_ratio_r2=float(1-np.square(yf-pf).sum()/denominator)
                if len(yf) > 1 and denominator > 0 else np.nan,
                valid=int(mask.sum()), total=len(g),
                true_nonfinite=int((~np.isfinite(y)).sum()),
                pred_nonfinite=int((~np.isfinite(p)).sum()))


def validate(rows, reference):
    keys = ['source', 'original_index', 'frame']
    if rows.duplicated(keys).any():
        raise ValueError('Duplicate network/frame rows')
    if set(rows.source) != SOURCES or set(rows.frame) != FRAMES:
        raise ValueError('Incomplete source/horizon coverage')
    joined = rows.merge(reference[keys + ['true_p_ratio']], on=keys,
                        suffixes=('', '_reference'), validate='one_to_one')
    if len(joined) != 400 or len(rows) != 400:
        raise ValueError('Wrong validation cohort: expected 20 networks/source/horizon')
    if not np.allclose(joined.true_p_ratio, joined.true_p_ratio_reference,
                       atol=1e-4, rtol=0, equal_nan=True):
        raise ValueError('Physical target mismatch with audited reconstruction')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = RESULTS / 'compact_lj_pratio/collection'
    original = pd.read_csv(base / 'per_network_rows.csv')
    records, statuses, inputs, reconstructions = [], [], {}, {}
    inputs[str(base / 'per_network_rows.csv')] = sha(base / 'per_network_rows.csv')
    reference = original[(original.variant == 'mp2_r16_d4') & (original.seed == 123)]

    def add(rows, variant, strategy, seed, kind):
        validate(rows, reference)
        for (source, frame), group in rows.groupby(['source', 'frame']):
            records.append(dict(variant=variant, strategy=strategy, seed=int(seed),
                                source=source, frame=int(frame), kind=kind, **score(group)))

    for seed in sorted(SEEDS):
        rows = original[(original.variant == 'mp2_r16_d4') & (original.seed == seed)]
        add(rows, 'mp2_r16_d4', 'ae', seed, 'reconstruction')
        reconstructions[('mp2_r16_d4', seed)] = rows
        candidates = [p for p in (RESULTS / 'compact_lj_mp2d/pratio').glob(
            f'mp2_r16_d2_s{seed}_code_v*/per_network_rows.csv')
            if (p.parent / 'completed.json').exists()]
        if len(candidates) > 1:
            raise ValueError(f'Multiple completed MP2D evaluations for seed {seed}')
        path = candidates[0] if candidates else RESULTS / 'compact_lj_mp2d/missing.csv'
        status = dict(variant='mp2_r16_d2', strategy='ae', seed=seed)
        if path.exists() and (path.parent / 'completed.json').exists():
            try:
                rows = pd.read_csv(path)
                add(rows, 'mp2_r16_d2', 'ae', seed, 'reconstruction')
                reconstructions[('mp2_r16_d2', seed)] = rows
                inputs[str(path)] = sha(path)
                status['status'] = 'completed'
            except Exception as exc:
                status.update(status='invalid', error=str(exc))
        else:
            status['status'] = 'pending'
        statuses.append(status)

    for dimension in (2, 4):
        variant = f'mp2_r16_d{dimension}'
        for strategy in ('one_step', 'multistep8'):
            for seed in sorted(SEEDS):
                status = dict(variant=variant, strategy=strategy, seed=seed)
                candidates = []
                for path in (OUT / 'runs').glob('*/per_network_rows.csv'):
                    if 'smoke' in path.parent.name:
                        continue
                    if not (path.parent / 'completed.json').exists():
                        continue
                    head = pd.read_csv(path, nrows=1)
                    if len(head) and head.variant.iloc[0] == variant and head.strategy.iloc[0] == strategy and int(head.seed.iloc[0]) == seed:
                        candidates.append(path)
                if not candidates:
                    failure = OUT / 'runs' / f'{variant}_{strategy}_s{seed}' / 'failed.json'
                    status['status'] = 'failed' if failure.exists() else 'pending'
                    if failure.exists():
                        status['error'] = json.loads(failure.read_text()).get('error')
                else:
                    try:
                        if len(candidates) != 1:
                            raise ValueError('Multiple completed runs for one intended cell')
                        path = candidates[0]
                        rows = pd.read_csv(path)
                        if set(rows.kind) != {'autonomous', 'reconstruction'}:
                            raise ValueError('Missing paired reconstruction/autonomous rows')
                        reconstructed = rows[rows.kind == 'reconstruction']
                        validate(reconstructed, reference)
                        if (variant, seed) not in reconstructions:
                            raise ValueError('Independent reconstruction evaluation is still pending')
                        frozen = reconstructions[(variant, seed)]
                        check = reconstructed.merge(frozen, on=['source', 'original_index', 'frame'],
                            suffixes=('', '_frozen'), validate='one_to_one')
                        if not np.allclose(check.pred_p_ratio, check.pred_p_ratio_frozen,
                                           atol=1e-4, rtol=0, equal_nan=True):
                            raise ValueError('Frozen AE reconstruction changed during dynamics fitting')
                        if set(rows.checkpoint_sha256) != set(frozen.checkpoint_sha256):
                            raise ValueError('Wrong frozen AE checkpoint hash')
                        add(rows[rows.kind == 'autonomous'], variant, strategy, seed, 'autonomous')
                        inputs[str(path)] = sha(path)
                        status['status'] = 'completed'
                    except Exception as exc:
                        status.update(status='invalid', error=str(exc))
                statuses.append(status)

    table = pd.DataFrame(records)
    keys = ['variant', 'strategy', 'source', 'frame', 'kind']
    table.to_csv(OUT / 'response_seed_results.csv', index=False)
    aggregate = table.groupby(keys).agg(r2_mean=('p_ratio_r2', 'mean'),
        r2_std=('p_ratio_r2', 'std'), seeds=('seed', 'nunique'),
        finite_r2_seeds=('p_ratio_r2', 'count'), valid=('valid', 'sum'),
        total=('total', 'sum')).reset_index()
    aggregate['target_r2'] = .4
    aggregate['mean_meets_target'] = aggregate.r2_mean.ge(.4)
    aggregate.to_csv(OUT / 'response_aggregate.csv', index=False)
    pd.DataFrame(statuses).to_csv(OUT / 'run_status.csv', index=False)
    metadata = dict(expected_new_ae=5, expected_propagators=20,
        status_counts=pd.Series([s['status'] for s in statuses]).value_counts().to_dict(),
        complete=all(s['status'] == 'completed' for s in statuses),
        metric='physical-coordinate endpoint directional-side p-ratio R2',
        count_qualification='20 unique validation networks/source repeated across five model seeds',
        final_test_used=False, checkpoint_selection='state loss only; target 0.4 is a reporting goal',
        collector_sha256=sha(Path(__file__)), input_sha256=inputs)
    (OUT / 'collection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lines = ['# Reconstruction and autonomous response results', '',
        f"Complete: {metadata['complete']}. New-run statuses: {metadata['status_counts']}.", '',
        'Physical endpoint p-ratio R²; seed mean ± sample SD. Each seed uses the same',
        '20 validation networks/source. Mixed-T is post-fit transfer; final test is untouched.',
        'The 0.4 target is a reporting goal, never a training or checkpoint criterion.', '',
        '| Model | Strategy | Source | Step | R² mean ± SD | Seeds | Valid/total |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in aggregate.itertuples():
        lines.append(f'| {r.variant} | {r.strategy} | {r.source} | {r.frame} | '
                     f'{r.r2_mean:.3f} ± {r.r2_std:.3f} | {r.seeds} | {r.valid}/{r.total} |')
    (OUT / 'review.md').write_text('\n'.join(lines) + '\n')
    if metadata['complete']:
        marker = '### Completed matched MP2D/4D autonomous response comparison'
        entry = ('\n' + marker + '\n\n'
            'All five new MP2D reconstruction evaluations and twenty frozen-AE propagator\n'
            'runs passed collection. Physical target cohorts, frozen AE reconstruction\n'
            'predictions and checkpoint hashes matched the independent reconstruction\n'
            'evaluations. Source-wise R² versus step, five-seed SD and valid/total counts:\n'
            '`notebooks/results/compact_lj_response_rollout/review.md`; exact rows,\n'
            'recipes, hashes, failures and job identities remain alongside the review.\n'
            'No response-selected checkpoints or final-test data were used.\n')
        for name in ('experiment_results_index.md', '06b_experiment_log.md'):
            path = ROOT / 'notebooks/latent_space' / name
            if marker not in path.read_text():
                with path.open('a') as handle:
                    handle.write(entry)
    print(json.dumps({k: metadata[k] for k in ('complete', 'status_counts')}, indent=2))


if __name__ == '__main__':
    main()
