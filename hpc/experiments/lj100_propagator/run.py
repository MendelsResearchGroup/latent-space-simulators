#!/usr/bin/env python3
"""Fine-tune shared latent propagation through100 steps; keep AE frozen."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ['LSS_PROJECT_ROOT']).resolve()
CODE = Path(os.environ['LSS_CODE_ROOT']).resolve()
sys.path.insert(0, str(CODE / 'src'))
from lss.dynamics.experiment import _load_ae_cache
from lss.dynamics.propagator import make_latent_propagator
from lss.dynamics.training import (
    LatentNormalizer, decode_latent_positions, encode_frame_latent,
    encode_reference_context, latent_step,
)
from evaluate import evaluate

OUT = ROOT / 'notebooks/results/lj100_propagator'
NAMES = ('reid', 'depablo_low_temp', 'lj_noisy')
SUP = (1, 2, 3, 5, 10, 25, 50, 75, 100)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def digest(tensors):
    result = hashlib.sha256()
    for key in sorted(tensors):
        result.update(key.encode())
        result.update(tensors[key].detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def verify_code():
    manifest = json.loads((CODE / 'manifest.json').read_text())
    assert {key: sha(CODE / key) for key in manifest} == manifest


def cohort(sims, recipe, split, smoke):
    result = []
    for spec in recipe['source']['dataset_mixture']:
        rows = [sim for sim in sims if str(sim[0].source_name) == spec['name']]
        ids = spec['split_indices'][split]
        assert len(rows) == len(ids)
        selected = list(zip(rows, ids))
        if smoke:
            selected = selected[:2 if split == 'train' else 1]
        result.extend((sim, spec['name'], int(index)) for sim, index in selected)
    assert {row[1] for row in result} == set(NAMES)
    return result


@torch.no_grad()
def cache(ae, rows, params, normalizers):
    result = []
    for sim, name, index in rows:
        true_z = torch.stack([
            encode_frame_latent(ae, sim, frame, pos_dim=2,
                node_feature_mode=params['node_feature_mode'],
                normalizers=normalizers, device='cpu')
            for frame in range(101)
        ]).detach()
        context = encode_reference_context(ae, [sim[0]], pos_dim=2,
            normalizers=normalizers, device='cpu', pool_mode='mean').detach()
        result.append((sim, true_z, context, name, index))
    return result


def rollout(dyn, initial_z, context, stats):
    result = [initial_z]
    for _ in range(100):
        result.append(latent_step(dyn, result[-1], stats,
                                  loss_mode='delta', context=context))
    return result


def losses(ae, dyn, rows, stats, normalizers):
    latent_losses, position_losses = [], []
    for sim, true_z, context, _, _ in rows:
        pred = rollout(dyn, true_z[0], context, stats)
        reference = sim[0].x[:, :2].float()
        span = (reference.amax(0) - reference.amin(0)).clamp_min(1e-6)
        scale = span * normalizers['target_std'].reshape(-1)
        latent, position = [], []
        for frame in SUP:
            latent.append(((pred[frame] - true_z[frame]) / stats.z_std.squeeze(0)).square().mean())
            decoded = decode_latent_positions(ae, [sim[0]], pred[frame], 0,
                pos_dim=2, ae_target_mode='normalized_delta',
                normalizers=normalizers, device='cpu')
            position.append(((decoded - sim[frame].x[:, :2].float()) / scale).square().mean())
        latent_losses.append(torch.stack(latent).mean())
        position_losses.append(torch.stack(position).mean())
    return torch.stack(latent_losses).mean(), torch.stack(position_losses).mean()


def train_epoch(ae, dyn, data, stats, normalizers, objective, optimizer, rng):
    dyn.train()
    groups = {name: [row for row in data if row[3] == name] for name in NAMES}
    for group in groups.values():
        rng.shuffle(group)
    steps = max(len(groups['reid']), len(groups['depablo_low_temp']), (len(groups['lj_noisy']) + 1) // 2)
    logs = {name: [] for name in NAMES}
    for step in range(steps):
        # Each source contributes exactly one third of each optimizer objective.
        batches = [[groups[name][step % len(groups[name])]] for name in NAMES[:2]]
        batches.append([groups['lj_noisy'][(2 * step + offset) % len(groups['lj_noisy'])]
                        for offset in range(2)])
        source_losses = []
        for name, batch in zip(NAMES, batches):
            latent, position = losses(ae, dyn, batch, stats, normalizers)
            value = latent if objective == 'latent_state' else position + 0.1 * latent
            source_losses.append(value)
            logs[name].append(float(value.detach()))
        optimizer.zero_grad(set_to_none=True)
        value = torch.stack(source_losses).mean()
        assert torch.isfinite(value), 'Nonfinite training loss'
        value.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    return {f'train_{name}': float(np.mean(values)) for name, values in logs.items()}


@torch.no_grad()
def validate(ae, dyn, data, stats, normalizers):
    dyn.eval()
    logs = {}
    for name in NAMES:
        values = [losses(ae, dyn, [row], stats, normalizers) for row in data if row[3] == name]
        logs[f'val_latent_{name}'] = float(np.mean([float(v[0]) for v in values]))
        logs[f'val_position_{name}'] = float(np.mean([float(v[1]) for v in values]))
    logs['val_latent_state'] = float(np.mean([logs[f'val_latent_{name}'] for name in NAMES]))
    logs['val_decoded_position'] = float(np.mean([logs[f'val_position_{name}'] for name in NAMES]))
    assert np.isfinite(list(logs.values())).all(), 'Nonfinite validation loss'
    return logs


def baseline_parity(ae, dyn, stats, data, params, normalizers, seed, objective, lr, baseline_dir, out):
    selected = [next(row for row in data if row[3] == name) for name in NAMES]
    table = evaluate(ae, dyn, stats, selected, params, normalizers,
                     seed, objective, lr, (25, 50, 100))
    old = pd.read_csv(baseline_dir / 'per_network_rows.csv')
    keys = ['source', 'original_index', 'frame', 'kind']
    matched = table.merge(old, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    assert len(matched) == 18
    differences = {}
    for column in ('true_p_ratio', 'pred_p_ratio'):
        a, b = matched[column + '_new'], matched[column + '_old']
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        differences[column] = float(np.max(np.abs(a - b)))
    table.to_csv(out / 'initial_parity_rows.csv', index=False)
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(123, 456, 786), required=True)
    parser.add_argument('--objective', choices=('latent_state', 'decoded_positions'), required=True)
    parser.add_argument('--lr', type=float, choices=(1e-4, 3e-4), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    verify_code()
    start = time.time()
    name = f'mp2d_s{args.seed}_{args.objective}_lr{args.lr:g}'
    if args.smoke:
        name += '_smoke_' + CODE.name
    out = OUT / 'runs' / name
    out.mkdir(parents=True, exist_ok=False)
    try:
        ae_dir = ROOT / 'notebooks/results/compact_lj_mp2d' / f'mp2_r16_d2_s{args.seed}_code_v4'
        baseline_dir = ROOT / 'notebooks/results/compact_lj_response_rollout/runs' / f'mp2_r16_d2_one_step_s{args.seed}'
        ae_file, init_file = ae_dir / 'ae.pt', baseline_dir / 'propagator_bundle.pt'
        recipe = json.loads((ae_dir / 'recipe.json').read_text())
        initial_recipe = json.loads((baseline_dir / 'recipe.json').read_text())
        assert not recipe['response_selection'] and not recipe['final_test_used']
        assert not initial_recipe['response_selection'] and not initial_recipe['final_test_used']
        assert json.loads((ae_dir / 'completed.json').read_text())['ae_file_sha256'] == sha(ae_file)
        assert json.loads((baseline_dir / 'completed.json').read_text())['status'] == 'completed'
        protocol = json.loads((CODE / 'protocol.json').read_text())
        assert protocol['source'] == recipe['source']
        planned = dict(vars(args), protocol=protocol, protocol_sha256=sha(CODE / 'protocol.json'),
            ae_sha256=sha(ae_file), init_sha256=sha(init_file), ae_recipe_sha256=sha(ae_dir / 'recipe.json'),
            initial_recipe_sha256=sha(baseline_dir / 'recipe.json'), dataset_sha256=recipe['dataset_sha256'],
            code_root=str(CODE), code_manifest_sha256=sha(CODE / 'manifest.json'),
            job_id=os.environ['PBS_JOBID'], max_epochs=1 if args.smoke else 40,
            patience=1 if args.smoke else 8, response_selection=False, final_test_used=False)
        (out / 'recipe.json').write_text(json.dumps(planned, indent=2) + '\n')
        result = _load_ae_cache(ae_file, recipe['config'], device='cpu')
        assert not result['test_data']
        ae, normalizers = result['ae'], result['normalizers']
        ae.eval()
        for parameter in ae.parameters():
            parameter.requires_grad_(False)
        bundle = torch.load(init_file, map_location='cpu', weights_only=False)
        keys = ('z_mean', 'z_std', 'dz_mean', 'dz_std', 'z_next_mean', 'z_next_std', 'context_mean', 'context_std')
        stats = LatentNormalizer(**{key: bundle['stats'][key] for key in keys})
        before = dict(ae=digest(ae.state_dict()), normalizers=digest(normalizers), latent_stats=digest(stats.as_dict()))
        train = cache(ae, cohort(result['train_data'], recipe, 'train', args.smoke), result['params'], normalizers)
        val = cache(ae, cohort(result['val_data'], recipe, 'val', args.smoke), result['params'], normalizers)
        dyn = make_latent_propagator(2, 64, model_type='delta_mlp', context_dim=16,
                                    graph_context_dim=16, context_pool_mode='mean')
        dyn.load_state_dict(bundle['dyn_state_dict'])
        parity = baseline_parity(ae, dyn, stats, val, result['params'], normalizers,
            args.seed, args.objective, args.lr, baseline_dir, out)
        optimizer = torch.optim.Adam(dyn.parameters(), lr=args.lr, weight_decay=1e-5)
        history = [dict(epoch=0, **validate(ae, dyn, val, stats, normalizers))]
        best = history[0]['val_decoded_position']
        state, best_epoch, bad = copy.deepcopy(dyn.state_dict()), 0, 0
        rng = random.Random(args.seed)
        for epoch in range(1, planned['max_epochs'] + 1):
            epoch_start = time.time()
            train_log = train_epoch(ae, dyn, train, stats, normalizers, args.objective, optimizer, rng)
            val_log = validate(ae, dyn, val, stats, normalizers)
            history.append(dict(epoch=epoch, seconds=time.time()-epoch_start, **train_log, **val_log))
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            print(json.dumps(history[-1]), flush=True)
            if val_log['val_decoded_position'] < best - 1e-5:
                best, state, best_epoch, bad = val_log['val_decoded_position'], copy.deepcopy(dyn.state_dict()), epoch, 0
            else:
                bad += 1
            if bad >= planned['patience']:
                break
        dyn.load_state_dict(state)
        gradient = 'not measured outside smoke'
        if args.smoke and args.objective == 'decoded_positions':
            optimizer.zero_grad(set_to_none=True)
            _, position = losses(ae, dyn, train[:1], stats, normalizers)
            position.backward()
            gradient = float(sum(p.grad.abs().sum() for p in dyn.parameters() if p.grad is not None))
            assert np.isfinite(gradient) and gradient > 0
        after = dict(ae=digest(ae.state_dict()), normalizers=digest(normalizers), latent_stats=digest(stats.as_dict()))
        assert before == after
        assert sha(ae_file) == planned['ae_sha256'] and sha(init_file) == planned['init_sha256']
        rows = evaluate(ae, dyn, stats, val, result['params'], normalizers,
                        args.seed, args.objective, args.lr, tuple(range(1, 101)))
        assert len(rows) == len(val) * 200
        rows.to_csv(out / 'per_network_rows.csv', index=False)
        torch.save(dict(dyn_state_dict=dyn.state_dict(), latent_stats=stats.as_dict(),
                        initial_one_step_sha256=planned['init_sha256']), out / 'propagator.pt')
        planned.update(frozen_before=before, frozen_after=after, baseline_parity=parity,
            pure_position_gradient=gradient, best_epoch=best_epoch, epochs_trained=len(history)-1,
            best_val_decoded_position=best, checkpoint_sha256=sha(out / 'propagator.pt'), seconds=time.time()-start)
        (out / 'recipe.json').write_text(json.dumps(planned, indent=2) + '\n')
        (out / 'completed.json').write_text(json.dumps(dict(status='completed', rows=len(rows),
            seconds=time.time()-start, job_id=os.environ['PBS_JOBID']), indent=2) + '\n')
    except Exception as error:
        (out / 'failed.json').write_text(json.dumps(dict(status='failed', error=f'{type(error).__name__}: {error}',
            job_id=os.environ['PBS_JOBID']), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
