#!/usr/bin/env python3
"""Seed-123 sparse subset of the frozen LJ latent-influence candidate union."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ['LSS_PROJECT_ROOT']).resolve()
FROZEN = ROOT / 'notebooks/results/lj_latent_influence_edges/code_v2'
OUT = ROOT / 'notebooks/results/lj_sparse_influence'
os.environ['LSS_CODE_ROOT'] = str(FROZEN)
spec = importlib.util.spec_from_file_location('frozen_lj_influence_run', FROZEN / 'run.py')
assert spec and spec.loader
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

TRAIN_COUNT, VALIDATION_COUNT = 60, 20
TOPK, RETAIN_FRACTION = 4, 0.10

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def sparse_subset(score, base):
    """Top-4 symmetric-union candidates, globally rank and retain ceil(10%)."""
    candidates = old.select_topk_nonphysical_edges(score, base, TOPK).cpu()
    values = [(float(score[i, j]), int(i), int(j)) for i, j in candidates.t().tolist()]
    values.sort(key=lambda row: (-row[0], row[1], row[2]))
    keep = math.ceil(RETAIN_FRACTION * len(values))
    selected = values[:keep]
    assert keep > 0 and len({(i, j) for _, i, j in selected}) == keep
    edge = torch.tensor([(i, j) for _, i, j in selected], dtype=torch.long).t().contiguous()
    return candidates, values, edge

def prepare_records(data, ae, latent_model, arm):
    records, selection = {'train': [], 'validation': []}, []
    for split, expected in (('train', TRAIN_COUNT), ('validation', VALIDATION_COUNT)):
        rows = [row for row in data['train' if split == 'train' else 'validation'] if row['source'] == 'lj_noisy']
        assert len(rows) == expected
        for row in rows:
            positions = old.normalized_positions(row)
            torch.testing.assert_close(positions[0], row['ref_pos'], rtol=0, atol=2e-6)
            base, base_stiffness = old.base_relation(row)
            score = old.latent_score(ae, latent_model, row, data)
            candidates, ranked, latent_edges = sparse_subset(score, base)
            generator = torch.Generator().manual_seed(123_000 + int(row['original_index']))
            random_edges = old.random_nonphysical_edges(len(positions[0]), base, latent_edges.shape[1], generator=generator)
            extra = latent_edges if arm == 'latent_sparse' else random_edges
            index = torch.cat((base, extra), 1)
            stiffness = torch.cat((base_stiffness, torch.zeros(extra.shape[1])))
            is_base = torch.cat((torch.ones(base.shape[1]), torch.zeros(extra.shape[1])))
            records[split].append(dict(original_index=int(row['original_index']), positions=positions, row=row,
                edge_index=index, stiffness=stiffness, is_base=is_base, is_added=1.0-is_base,
                base_edge_count=int(base.shape[1]), added_edge_count=int(extra.shape[1]),
                candidate_edge_count=int(candidates.shape[1]), candidate_edges=candidates,
                sparse_latent_edges=latent_edges, random_edges=random_edges))
            selected_pairs = {(int(i), int(j)) for i, j in latent_edges.t().tolist()}
            for rank, (value, i, j) in enumerate(ranked, 1):
                selection.append(dict(split=split, original_index=int(row['original_index']), candidate_rank=rank,
                    source_node=i, target_node=j, latent_score=value, retained_sparse=(i, j) in selected_pairs,
                    candidate_edge_count=len(ranked), retained_edge_count=len(selected_pairs)))
    return records['train'], records['validation'], pd.DataFrame(selection)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--arm', choices=('latent_sparse','random_sparse'), required=True); p.add_argument('--seed', type=int, choices=(123,), required=True); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    protocol = json.loads((FROZEN / 'protocol.json').read_text())
    cache = ROOT / protocol['frozen_inputs']['cache']; latent_path = ROOT / protocol['frozen_inputs']['latent_propagator']
    assert sha(cache) == protocol['frozen_inputs']['cache_sha256'] and sha(latent_path) == protocol['frozen_inputs']['latent_propagator_sha256']
    assert {n: sha(FROZEN / n) for n in json.loads((FROZEN / 'manifest.json').read_text())} == json.loads((FROZEN / 'manifest.json').read_text())
    data = torch.load(cache, map_location='cpu', weights_only=False)
    assert sha(Path(data['ae_checkpoint'])) == protocol['frozen_inputs']['autoencoder_sha256']
    ae = old.load_ae(data)
    latent_model = old.make_trajectory_propagator('history_attention', 'mean')
    latent_model.load_state_dict(torch.load(latent_path, map_location='cpu', weights_only=False)['state_dict'], strict=True); latent_model.eval()
    for parameter in latent_model.parameters(): parameter.requires_grad_(False)
    frozen_before = {'ae': old.state_digest(ae.state_dict()), 'latent': old.state_digest(latent_model.state_dict())}
    output = OUT / 'runs' / f'{a.arm}_s{a.seed}'; output.mkdir(parents=True, exist_ok=False)
    recipe = dict(study='lj_sparse_influence', arm=a.arm, seed=a.seed, job_id=os.environ.get('PBS_JOBID','local'),
        frozen_runner=str(FROZEN.relative_to(ROOT) / 'run.py'), frozen_runner_sha256=sha(FROZEN/'run.py'),
        frozen_protocol_sha256=sha(FROZEN/'protocol.json'), frozen_manifest_sha256=sha(FROZEN/'manifest.json'),
        sparse_runner_sha256=sha(Path(__file__)),
        selection='original top4 symmetric-union candidates from frozen latent_score; globally strongest ceil(0.10*M), ties canonical (i,j); no per-node minimum',
        influence='frozen global +/-0.001 normalized-latent propagation/decode response Gram score at horizons5/10/20',
        train_count=60, validation_count=20, architecture='unchanged frozen NormalizedAccelerationGNN hidden64 steps4',
        normalization='same frozen training-only per-arm normalizer fitting procedure', response_supervision_or_selection=False, final_test_used=False, frozen_before=frozen_before)
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    try:
        train_records, validation_records, selections = prepare_records(data, ae, latent_model, a.arm)
        selections.to_csv(output/'candidate_edge_scores.csv', index=False)
        chosen = []
        for split, records in (('train',train_records),('validation',validation_records)):
            for rec in records:
                sparse = {(int(i),int(j)) for i,j in rec['sparse_latent_edges'].t().tolist()}; random_edges = {(int(i),int(j)) for i,j in rec['random_edges'].t().tolist()}; candidates = {(int(i),int(j)) for i,j in rec['candidate_edges'].t().tolist()}
                chosen.extend(dict(split=split, original_index=rec['original_index'], arm=a.arm, selection=name, source_node=i,target_node=j,
                    candidate_edge_count=rec['candidate_edge_count'], retained_edge_count=rec['added_edge_count'], old_top4_candidate_overlap=1.0 if (i,j) in candidates else 0.0)
                    for name,pairs in (('latent_sparse',sparse),('random_sparse',random_edges)) for i,j in sorted(pairs))
        pd.DataFrame(chosen).to_csv(output/'selected_edges.csv',index=False)
        edge_stats = []
        for split, records in (('train', train_records), ('validation', validation_records)):
            for rec in records:
                degree = torch.zeros(len(rec['positions'][0]), dtype=torch.long)
                degree.index_add_(0, rec['edge_index'][0, rec['base_edge_count']:], torch.ones(rec['added_edge_count'], dtype=torch.long))
                degree.index_add_(0, rec['edge_index'][1, rec['base_edge_count']:], torch.ones(rec['added_edge_count'], dtype=torch.long))
                edge_stats.append(dict(split=split, original_index=rec['original_index'], candidate_edge_count=rec['candidate_edge_count'], added_edge_count=rec['added_edge_count'], retained_fraction=rec['added_edge_count']/rec['candidate_edge_count'], zero_added_edge_node_fraction=float((degree == 0).float().mean())))
        pd.DataFrame(edge_stats).to_csv(output/'selection_edge_stats.csv', index=False)
        normalizers = old.fit_normalizers(train_records)
        model = old.NormalizedAccelerationGNN(hidden_dim=64, steps=4)
        state, history = old.train(model, train_records, validation_records, normalizers, False, a.seed)
        model.load_state_dict(state); predictions = old.batched_rollout(model, validation_records, normalizers)
        torch.save({'original_indices':[record['original_index'] for record in validation_records],
                    'normalized_prediction_paths':predictions,
                    'normalized_truth_paths':[record['positions'] for record in validation_records],
                    'physical_half':[record['row']['half'] for record in validation_records],
                    'physical_center':[record['row']['center'] for record in validation_records]}, output/'predicted_position_paths.pt')
        table = old.evaluate_rollout(validation_records,predictions,a.arm,a.seed); summary = old.summarize(table)
        pd.DataFrame(history).to_csv(output/'history.csv',index=False); table.to_csv(output/'per_network_rows.csv',index=False); summary.to_csv(output/'summary.csv',index=False)
        torch.save({'state_dict':state,'normalizers':normalizers},output/'model.pt')
        frozen_after={'ae':old.state_digest(ae.state_dict()),'latent':old.state_digest(latent_model.state_dict())}; assert frozen_before == frozen_after
        recipe.update(checkpoint_sha256=sha(output/'model.pt'), frozen_after=frozen_after, summary=summary.to_dict('records'))
        (output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n'); (output/'completed.json').write_text(json.dumps({'status':'completed','job_id':recipe['job_id']},indent=2)+'\n')
    except Exception as error:
        (output/'failed.json').write_text(json.dumps({'status':'failed','error':f'{type(error).__name__}: {error}'},indent=2)+'\n'); raise

if __name__ == '__main__': main()
