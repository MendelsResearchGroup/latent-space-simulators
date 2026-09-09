#!/usr/bin/env python3
"""Post-fit endpoint p-ratio evaluation of a frozen AE; never trains or selects."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('LSS_PROJECT_ROOT', Path.cwd())).resolve()
CODE = Path(os.environ.get('LSS_EVAL_CODE_ROOT', ROOT)).resolve()
sys.path.insert(0, str(CODE/'src'))
import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from graph_utils.box import Box
import graph_utils.pratio as metric_module
from lss.data import POSITION_NORMALIZATION
from lss.dynamics.experiment import _load_ae_cache, resolve_train_val_test
from lss.dynamics.training import clone_graph, decode_latent_to_graph, encode_frame_latent

FRAMES = (25, 50, 100, 150, 199)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def physical(graph, reference):
    out = clone_graph(graph).cpu()
    if str(getattr(reference, 'coordinate_normalization', '')).lower() != POSITION_NORMALIZATION:
        raise ValueError('Expected saved reference-box position normalization')
    # decode_latent_to_graph intentionally returns a minimal graph; all inverse
    # metadata must come from the untouched reference, not from its clone.
    half = reference.reference_box_half_extent.detach().cpu().double()[:2]
    center = reference.reference_box_center.detach().cpu().double()[:2]
    out.x = out.x.detach().cpu().double().clone()
    out.x[:, :2] = out.x[:, :2]*half + center
    box = out.box
    out.box = Box(float(box.x1*half[0]+center[0]), float(box.x2*half[0]+center[0]),
                  float(box.y1*half[1]+center[1]), float(box.y2*half[1]+center[1]),
                  float(getattr(box, 'z1', -.1)), float(getattr(box, 'z2', .1)))
    out.box_tensor = half*torch.as_tensor(graph.box_tensor, dtype=torch.float64)[:2]
    out.coordinate_normalization = 'physical'
    return out


def extents(graph, sides):
    pos = graph.x[:, :2].detach().cpu().numpy()
    return np.array([pos[sides['right'], 0].mean()-pos[sides['left'], 0].mean(),
                     pos[sides['top'], 1].mean()-pos[sides['bottom'], 1].mean()])


def evaluate(args):
    start = time.time()
    torch.set_num_threads(1)
    out, checkpoint, recipe_file = map(Path, (args.out, args.checkpoint, args.recipe))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completed.json').exists():
        raise FileExistsError(f'Already completed: {out}')
    recipe = json.loads(recipe_file.read_text())
    train_done = json.loads((checkpoint.parent/'completed.json').read_text())
    checkpoint_hash, recipe_hash = sha(checkpoint), sha(recipe_file)
    assert checkpoint_hash == train_done['ae_file_sha256']
    assert not recipe['final_test_used'] and not recipe['response_selection']
    study = checkpoint.parent.parent.name
    assert study in {'compact_lj_reconstruction', 'compact_lj_spatial_decoder'}
    # Frozen bundle contains flattened training params and its original source spec.
    result = _load_ae_cache(checkpoint, {}, device=torch.device('cpu'))
    params, ae, normalizers = result['params'], result['ae'], result['normalizers']
    ae.eval()
    assert all(not p.requires_grad for p in ae.parameters())
    specs = recipe['source']['dataset_mixture']
    loaded_names = {str(sim[0].source_name) for sim in result['val_data']}
    assert loaded_names == {s['name'] for s in specs}
    groups = []
    for spec in specs:
        sims = [sim for sim in result['val_data'] if str(sim[0].source_name) == spec['name']]
        ids = list(spec['split_indices']['val'])
        assert len(sims) == len(ids) == 20 and len(set(ids)) == 20
        groups.append(('validation', spec, sims, ids))
    if recipe.get('mixed_evaluation'):
        mixed = copy.deepcopy(recipe['mixed_evaluation'])
        mixed_params = {**params, **mixed}
        train, sims, test, _ = resolve_train_val_test(mixed, mixed_params, split_seed=params['split_seed'])
        assert not train and not test
        for spec in mixed['dataset_mixture']:
            selected = [sim for sim in sims if str(sim[0].source_name) == spec['name']]
            ids = list(spec['split_indices']['val'])
            assert len(selected) == len(ids) == 20
            groups.append(('mixed_transfer', spec, selected, ids))
    rows, schema = [], []
    with torch.no_grad():
        for split, spec, sims, ids in groups:
            source = spec['name']
            for original_index, sim in zip(ids, sims):
                ref = sim[0]
                assert len(sim) >= 200 and ref.edge_attr.shape[1] == 5
                if source == 'lj_noisy':
                    assert spec['lj_max_graph_distance'] == 3
                    added = ref.edge_attr[:, 4] > .5
                    assert added.any() and torch.all(ref.edge_attr[added, 3] == 0)
                else:
                    assert torch.all(ref.edge_attr[:, 4] == 0)
                pref = physical(ref, ref)
                # An independent physical reference copy was preserved before normalization.
                inverse_error = float((pref.x[:, :2]-ref.reference_context_positions.cpu().double()[:, :2]).abs().max())
                if inverse_error > 2e-5:
                    raise ValueError(f'Physical reference inversion mismatch: {inverse_error}')
                # Preserve exact original side memberships. Tiny inverse-rounding
                # errors can break ties among boundary nodes on regular LJ grids.
                pref.x[:, :2] = ref.reference_context_positions.cpu().double()[:, :2]
                sides = directional_side_indices_from_box(pref, quantile=.10)
                model_sides = directional_side_indices_from_box(ref, quantile=.10)
                initial_extent = extents(pref, sides)
                schema.append({'source': source, 'split': split, 'original_index': original_index,
                               'inverse_reference_max_abs_error': inverse_error,
                               'half_extent': ref.reference_box_half_extent.tolist(),
                               'edge_count': int(ref.edge_index.shape[1])})
                for frame in FRAMES:
                    z = encode_frame_latent(ae, sim, frame, pos_dim=2, node_feature_mode=params['node_feature_mode'], normalizers=normalizers, device='cpu')
                    pred = decode_latent_to_graph(ae, sim, z, frame, pos_dim=2, ae_target_mode=params['ae_target_mode'], normalizers=normalizers, device='cpu')
                    truth_phys, pred_phys = physical(sim[frame], ref), physical(pred, ref)
                    true_p = float(calc_p_ratio_rollout_sides([pref, truth_phys], -1, side_idx=sides))
                    pred_p = float(calc_p_ratio_rollout_sides([pref, pred_phys], -1, side_idx=sides))
                    model_true = float(calc_p_ratio_rollout_sides([ref, sim[frame]], -1, side_idx=model_sides))
                    model_pred = float(calc_p_ratio_rollout_sides([ref, pred], -1, side_idx=model_sides))
                    true_strain = (extents(truth_phys, sides)-initial_extent)/initial_extent
                    pred_strain = (extents(pred_phys, sides)-initial_extent)/initial_extent
                    rows.append(dict(study=study, variant=recipe['variant'], seed=recipe['seed'],
                        source=source, split=split, transfer_only=split=='mixed_transfer', original_index=int(original_index),
                        frame=frame, target_index=frame, true_p_ratio=true_p, pred_p_ratio=pred_p,
                        model_true_p_ratio=model_true, model_pred_p_ratio=model_pred,
                        finite_true=math.isfinite(true_p), finite_pred=math.isfinite(pred_p),
                        finite_pair=math.isfinite(true_p) and math.isfinite(pred_p),
                        true_strain_x=true_strain[0], true_strain_y=true_strain[1],
                        pred_strain_x=pred_strain[0], pred_strain_y=pred_strain[1],
                        near_zero_strain_true=bool(np.max(np.abs(true_strain)) < 1e-5),
                        near_zero_strain_pred=bool(np.max(np.abs(pred_strain)) < 1e-5),
                        coordinate_mse=float((pred.x[:, :2].cpu()-sim[frame].x[:, :2].cpu()).square().mean()),
                        checkpoint_sha256=checkpoint_hash, recipe_sha256=recipe_hash,
                        data_sha256=recipe['dataset_sha256'][source], manifest_sha256=recipe['manifest_sha256'],
                        p_ratio_method='physical_endpoint_sides', model_p_ratio_method='model_coordinate_endpoint_sides'))
    table = pd.DataFrame(rows)
    original = pd.read_csv(checkpoint.parent/'source_frame_rows.csv')
    original = original[original.split.isin(['validation', 'mixed_transfer']) & original.frame.isin(FRAMES)]
    parity = table.merge(original, on=['source', 'split', 'original_index', 'frame'], suffixes=('_new', '_original'), validate='one_to_one')
    # Frame150 is a newly requested response horizon; historical coordinate
    # tables saved149 instead. Verify every common row, including100 and199.
    if len(parity) != len(original) or not {25, 50, 100, 199}.issubset(set(parity.frame)):
        raise ValueError(f'Coordinate parity coverage mismatch: new={len(table)}, original={len(original)}, matched={len(parity)}')
    max_delta = float((parity.coordinate_mse_new-parity.coordinate_mse_original).abs().max())
    if not math.isfinite(max_delta) or max_delta > 1e-10:
        raise ValueError(f'Coordinate parity failed: max absolute MSE delta {max_delta}')
    assert sha(checkpoint) == checkpoint_hash and sha(recipe_file) == recipe_hash
    table.to_csv(out/'per_network_rows.csv', index=False)
    (out/'recipe.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'training_recipe': str(recipe_file),
        'checkpoint_sha256': checkpoint_hash, 'training_recipe_sha256': recipe_hash,
        'runner_sha256': sha(Path(__file__)), 'metric_module_sha256': sha(Path(metric_module.__file__)),
        'source_code_root': str(CODE), 'torch_version': torch.__version__,
        'frames': FRAMES, 'metric': 'physical-coordinate endpoint sides, quantile0.10, eps1e-12',
        'near_zero_strain_flag': 'max(abs(relative_side_strains)) <1e-5; flag only, never filters',
        'final_test_used': False, 'training_performed': False, 'selection_performed': False,
        'schema_checks': schema, 'original_training_recipe': recipe}, indent=2)+'\n')
    (out/'completed.json').write_text(json.dumps({'status': 'completed', 'rows': len(table),
        'job_id': os.environ.get('PBS_JOBID'), 'seconds': time.time()-start,
        'checkpoint_sha256': checkpoint_hash, 'checkpoint_sha256_after': sha(checkpoint),
        'recipe_sha256': recipe_hash, 'max_coordinate_mse_delta': max_delta}, indent=2)+'\n')
    print(json.dumps({'status': 'completed', 'rows': len(table), 'max_coordinate_mse_delta': max_delta}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'recipe', 'out'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    try:
        evaluate(args)
    except Exception as exc:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out/'failed.json').write_text(json.dumps({'status': 'failed', 'job_id': os.environ.get('PBS_JOBID'),
            'error': f'{type(exc).__name__}: {exc}', 'runner_sha256': sha(Path(__file__))}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
