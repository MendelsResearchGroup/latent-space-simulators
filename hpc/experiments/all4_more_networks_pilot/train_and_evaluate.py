#!/usr/bin/env python3
"""Frozen-v5 AE-only 32-network coverage pilot; no response quantities enter fitting."""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path('/rg/mendels_prj/alexander.z/latent-space-simulators')
FROZEN = ROOT / 'notebooks/results/all4_coverage_1000/code_v5'
OUT = ROOT / 'notebooks/results/all4_more_networks_pilot'
sys.path[:0] = [str(FROZEN), str(FROZEN / 'src'), str(ROOT / 'src'), '/rg/mendels_prj/alexander.z/MetaForge/src']

from train import (VALIDATION_IDS, assert_historical_config, config, dump, fit_uniform,
                   seed_everything, sha)
from evaluate import evaluate_autoencoder
from lss.dynamics.experiment import resolve_train_val_test, run_latent_experiment

TRAIN_IDS = list(range(20)) + list(range(28, 40))
LOW_SOURCES = ('node_low', 'stiffness_low')
HELDOUT_IDS = list(range(40, 120))
HORIZONS = (2, 5, 10, 20, 33, 50, 99, 199)
PAIRS = (('node_low', 102), ('stiffness_low', 51), ('node_low', 47), ('stiffness_low', 106))


def source_spec(manifest):
    specs = []
    for name in ('node_low', 'node_mixed', 'stiffness_low', 'stiffness_mixed'):
        specs.append(dict(name=name, label=name, path=manifest['datasets'][name]['trainval_path'],
            train_count=len(TRAIN_IDS), val_count=len(VALIDATION_IDS[name]),
            split_indices=dict(train=TRAIN_IDS, val=VALIDATION_IDS[name], test=[])))
    return dict(name='mini_shared', dataset_name='mini_shared', source_name='mini_shared',
        label='mini_shared', path=specs[0]['path'], dataset_mixture=specs,
        coordinate_normalization='position_normalization')


def source_test_spec(name, manifest, params):
    entry = manifest['datasets'][name]
    assert sha(entry['test_path']) == entry['test_sha256']
    spec = dict(name=name, label=name, path=entry['test_path'], train_count=0, val_count=80,
        split_indices=dict(train=[], val=HELDOUT_IDS, test=[]))
    source = dict(name=name, label=name, path=entry['test_path'], dataset_name=name,
        source_name=name, coordinate_normalization='position_normalization', dataset_mixture=[spec])
    _, sims, _, _ = resolve_train_val_test(source, params, split_seed=123)
    assert len(sims) == 80
    for index, sim in enumerate(sims):
        assert len(sim) == 200 and all(graph.time == frame * 1000 for frame, graph in enumerate(sim))
        for graph in sim:
            graph.source_sim_id = index + 40  # resolver indexes cached test data 0..79
    assert [sim[0].source_sim_id for sim in sims] == HELDOUT_IDS
    return sims


def write_comparisons(table):
    baseline = ROOT / 'notebooks/results/all4_coverage_1000/all4_twenty_each_retry_v5_validation30/test/per_network.csv'
    old = pd.read_csv(baseline)
    old = old.loc[old['mode'].eq('ae_reconstruction')].copy()
    new = table.copy()
    new.to_csv(OUT / 'test/per_network.csv', index=False)
    new_steps = []
    for name in LOW_SOURCES:
        path = OUT / 'test' / name / 'per_step.csv'
        new_steps.append(pd.read_csv(path))
    steps = pd.concat(new_steps, ignore_index=True)
    steps.to_csv(OUT / 'test/per_step.csv', index=False)
    selected = steps.loc[steps.frame.isin(HORIZONS), ['source', 'frame', 'p_ratio_r2', 'p_ratio_valid', 'p_ratio_total']]
    selected.to_csv(OUT / 'horizon_source_pratio_r2.csv', index=False)
    records = []
    for source, trajectory_id in PAIRS:
        for label, data in [('new32', new), ('cached20', old)]:
            subset = data.loc[(data.source == source) & (data.trajectory_id == trajectory_id) & data.frame.between(1, 199)].copy()
            assert len(subset) == 199
            subset['model'] = label
            subset['abs_p_ratio_error'] = (subset.pred_p_ratio - subset.true_p_ratio).abs()
            records.append(subset[['model', 'source', 'trajectory_id', 'frame', 'true_p_ratio', 'pred_p_ratio', 'abs_p_ratio_error', 'p_ratio_valid']])
    curves = pd.concat(records, ignore_index=True)
    curves.to_csv(OUT / 'pair_pratio_curves.csv', index=False)
    summary = (curves.loc[curves.frame.between(1, 99)].groupby(['model', 'source', 'trajectory_id'], as_index=False)
        .agg(curve_mae=('abs_p_ratio_error', 'mean'), valid=('p_ratio_valid', 'sum'), total=('p_ratio_valid', 'size')))
    summary.to_csv(OUT / 'pair_curve_mae_frames1_99.csv', index=False)
    # Baseline per-step already records the exact same pooled scalar R2 and valid counts.
    old_steps = pd.concat([pd.read_csv(ROOT / 'notebooks/results/all4_coverage_1000/all4_twenty_each_retry_v5_validation30/test' / n / 'per_step.csv') for n in LOW_SOURCES])
    old_steps = old_steps.loc[old_steps['mode'].eq('ae_reconstruction')].copy()
    comparison = selected.merge(old_steps.loc[old_steps.frame.isin(HORIZONS), ['source', 'frame', 'p_ratio_r2', 'p_ratio_valid', 'p_ratio_total']], on=['source', 'frame'], suffixes=('_new32', '_cached20'))
    assert len(comparison) == len(selected) == len(LOW_SOURCES) * len(HORIZONS)
    comparison['p_ratio_r2_delta_new32_minus_cached20'] = comparison.p_ratio_r2_new32 - comparison.p_ratio_r2_cached20
    comparison.to_csv(OUT / 'horizon_vs_cached20.csv', index=False)


def main():
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    seed_everything(123)
    manifest = json.loads((FROZEN / 'data_manifest.json').read_text())
    for name, entry in manifest['datasets'].items():
        assert sha(entry['trainval_path']) == entry['trainval_sha256'], name
    cfg = config(OUT)
    assert_historical_config(cfg, OUT)
    assert cfg['should_train_propagator'] is False and cfg['should_rollout'] is False
    source = source_spec(manifest)
    assert set(TRAIN_IDS).isdisjoint(set(range(20, 28))) and max(TRAIN_IDS) == 39
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / 'recipe.json', dict(
        study='all4_more_networks_pilot', seed=123, job_id=os.environ['PBS_JOBID'],
        frozen_code_root=str(FROZEN.relative_to(ROOT)), frozen_code_manifest_sha256=sha(FROZEN / 'manifest.json'),
        frozen_data_manifest_sha256=sha(FROZEN / 'data_manifest.json'), config=cfg, source=source,
        train_ids=TRAIN_IDS, validation_ids_by_source=VALIDATION_IDS,
        heldout_evaluation_ids=HELDOUT_IDS, heldout_evaluation_sources=list(LOW_SOURCES),
        architecture='2D message_passing_reference16', ae_only=True,
        objective='frozen v5: warmup1 then uniform exactly-1-per-frame displacement MSE, max60/patience12',
        checkpoint_selection='validation displacement MSE only; no p-ratio/strain inputs, loss, or selection',
        evaluator='frozen evaluate_autoencoder physical_endpoint_fixed_frame0_sides_quantile_0.10, frames1..199',
        test_used_for_selection=False, final_test_ids_used=[]))
    result = run_latent_experiment(source, cfg, device='cpu')
    # The split resolver reindexes explicit noncontiguous selections; restore
    # immutable original dataset IDs before recording exposure or evaluation.
    for sims, ids_by_source in ((result['train_data'], {name: TRAIN_IDS for name in VALIDATION_IDS}),
                                (result['val_data'], VALIDATION_IDS)):
        for name, ids in ids_by_source.items():
            source_sims = [sim for sim in sims if str(sim[0].source_name) == name]
            assert len(source_sims) == len(ids)
            for sim, original_id in zip(source_sims, ids):
                for graph in sim:
                    graph.source_sim_id = original_id
    exposure = []
    for split, sims in [('train', result['train_data']), ('validation', result['val_data'])]:
        for sim in sims:
            graph = sim[0]
            exposure.append(dict(split=split, source=str(graph.source_name), trajectory_id=int(graph.source_sim_id), frames=len(sim)))
    pd.DataFrame(exposure).to_csv(OUT / 'exposure.csv', index=False)
    assert set(pd.DataFrame(exposure).query("split == 'train'").trajectory_id) == set(TRAIN_IDS)
    result['ae_history'].to_csv(OUT / 'ae_warmup_history.csv', index=False)
    ae_sha = fit_uniform(result, OUT, 60)
    dump(OUT / 'ae_completed.json', dict(status='completed', sha256=ae_sha, ae_only=True))
    all_rows = []
    for name in LOW_SOURCES:
        sims = source_test_spec(name, manifest, result['params'])
        rows = evaluate_autoencoder({**result, 'val_data': sims}, seed=123, latent_dim=2,
            output=OUT / 'test' / name, horizon=199, split='test')
        assert set(rows.trajectory_id) == set(HELDOUT_IDS) and set(rows.frame) == set(range(200))
        all_rows.append(rows)
        del sims
        gc.collect()
    all_rows = pd.concat(all_rows, ignore_index=True)
    write_comparisons(all_rows)
    dump(OUT / 'completed.json', dict(status='completed', job_id=os.environ['PBS_JOBID'], ae_sha256=ae_sha,
        ae_only=True, test_used_for_selection=False, evaluated_sources=list(LOW_SOURCES),
        evaluated_ids=HELDOUT_IDS, frames=list(range(1, 200))))


if __name__ == '__main__':
    main()
