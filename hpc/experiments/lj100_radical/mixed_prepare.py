#!/usr/bin/env python3
"""Cache frozen AE labels and reference embeddings once per model seed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(os.environ['LSS_PROJECT_ROOT']).resolve()
CODE = Path(os.environ['LSS_CODE_ROOT']).resolve()
sys.path.insert(0, str(CODE / 'src'))
from graph_utils.box import Box
from lss.graph import clone_graph
from lss.dynamics.experiment import resolve_train_val_test
from run import load_ae
from lss.dynamics.training import (encode_frame_latent, decode_latent_positions,
    reference_positions_for_model, compact_reference_edge_features)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            value.update(part)
    return value.hexdigest()


def digest(state):
    value = hashlib.sha256()
    for key in sorted(state):
        value.update(key.encode())
        value.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(123,456,786), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    start = time.time()
    manifest = json.loads((CODE / 'manifest.json').read_text())
    assert {key: sha(CODE / key) for key in manifest} == manifest
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists()
    ae_dir = ROOT / 'notebooks/results/compact_lj_mp2d' / f'mp2_r16_d2_s{args.seed}_code_v4'
    init_dir = ROOT / 'notebooks/results/compact_lj_response_rollout/runs' / f'mp2_r16_d2_one_step_s{args.seed}'
    checkpoint = ae_dir / 'ae.pt'
    recipe = json.loads((ae_dir / 'recipe.json').read_text())
    assert not recipe['response_selection'] and not recipe['final_test_used']
    assert json.loads((ae_dir / 'completed.json').read_text())['ae_file_sha256'] == sha(checkpoint)
    assert recipe['source'] == json.loads((CODE / 'protocol.json').read_text())['source']
    provenance = dict(seed=args.seed, smoke=args.smoke, job_id=os.environ['PBS_JOBID'],
        ae_sha256=sha(checkpoint), dataset_sha256=recipe['dataset_sha256'],
        ae_recipe_sha256=sha(ae_dir/'recipe.json'),
        latent_initialization_sha256=sha(init_dir/'propagator_bundle.pt'),
        code_manifest_sha256=sha(CODE/'manifest.json'), protocol_sha256=sha(CODE/'protocol.json'),
        source=recipe['source'], final_test_used=False, response_features=False)
    out.with_suffix('.recipe.json').write_text(json.dumps(provenance, indent=2)+'\n')
    training_cache = torch.load(ROOT/'notebooks/results/lj100_broad/cache'/f's{args.seed}.pt',map_location='cpu',weights_only=False)
    ae = load_ae(training_cache)
    bundle = torch.load(checkpoint,map_location='cpu',weights_only=False)
    mixed = recipe['mixed_evaluation']
    params = {**bundle['params'],**mixed}
    train,val,test,_ = resolve_train_val_test(mixed,params,split_seed=params['split_seed'])
    assert not train and not test and len(val)==20
    result = dict(ae=ae,normalizers=training_cache['normalizers'],params=params,train_data=train,val_data=val,test_data=test)
    recipe['source'] = mixed
    provenance['evaluation_source'] = mixed
    provenance['postfit_only'] = True
    assert not result['test_data']
    ae, norm, params = result['ae'], result['normalizers'], result['params']
    ae.eval()
    before = digest(ae.state_dict())
    init = torch.load(init_dir/'propagator_bundle.pt', map_location='cpu', weights_only=False)
    latent_stats = {key: init['stats'][key] for key in ('z_mean','z_std','dz_mean','dz_std','context_mean','context_std')}
    cache = dict(train=[], validation=[], normalizers=norm, latent_stats=latent_stats,
                 ae_checkpoint=str(checkpoint), provenance=provenance)
    max_error = 0.0
    with torch.no_grad():
        for split, source_data in [('train',result['train_data']),('validation',result['val_data'])]:
            for spec in recipe['source']['dataset_mixture']:
                sims = [sim for sim in source_data if str(sim[0].source_name)==spec['name']]
                indices = spec['split_indices']['train' if split=='train' else 'val']
                assert len(sims)==len(indices)
                pairs = list(zip(sims,indices))
                if args.smoke:
                    pairs = pairs[:2 if split=='train' else 1]
                for rank,(sim,index) in enumerate(pairs):
                    ref = sim[0]
                    reference = ref.x[:,:2].float()
                    span = (reference.amax(0)-reference.amin(0)).clamp_min(1e-6)
                    reference_edges = compact_reference_edge_features(ref, pos_dim=2, device='cpu')
                    normalized_edges = (reference_edges-norm['ref_edge_mean'])/norm['ref_edge_std']
                    h_ref = ae.encode_reference_graph(reference_positions_for_model(ref,pos_dim=2,device='cpu'),
                                                       normalized_edges,ref.edge_index.long()).detach()
                    assert h_ref.shape==(len(reference),16)
                    z = torch.stack([encode_frame_latent(ae,sim,f,pos_dim=2,
                        node_feature_mode=params['node_feature_mode'],normalizers=norm,device='cpu') for f in range(101)])
                    positions = torch.stack([frame.x[:,:2].float() for frame in sim[:101]])
                    half = ref.reference_box_half_extent.double()[:2]
                    center = ref.reference_box_center.double()[:2]
                    physical_positions = positions.double()*half+center
                    physical_ref = clone_graph(ref).cpu()
                    physical_ref.x = physical_ref.x.double().clone()
                    physical_ref.x[:,:2] = physical_positions[0]
                    box = ref.box
                    physical_ref.box = Box(float(box.x1*half[0]+center[0]),float(box.x2*half[0]+center[0]),
                        float(box.y1*half[1]+center[1]),float(box.y2*half[1]+center[1]),-.1,.1)
                    physical_ref.box_tensor = ref.box_tensor.double()[:2]*half
                    physical_ref.coordinate_normalization = 'physical'
                    record = dict(source=spec['name'],original_index=int(index),split=split,
                        latent_norm=((z-latent_stats['z_mean'])/latent_stats['z_std']).detach(),
                        h_ref=h_ref, reference_tokens=torch.cat((h_ref,reference),-1).detach(),
                        target_norm=(((positions-reference)/span-norm['target_mean'])/norm['target_std']).detach(),
                        physical_positions=physical_positions,physical_reference=physical_ref,
                        ref_pos=reference,span=span,half=half,center=center)
                    if split=='validation' and rank==0:
                        batch = torch.zeros(len(reference),dtype=torch.long)
                        for frame in (1,25,100):
                            decoded = ae.decode(z[frame].unsqueeze(0),h_ref,batch)
                            reconstructed = reference+(decoded*norm['target_std']+norm['target_mean'])*span
                            expected = decode_latent_positions(ae,[ref],z[frame],0,pos_dim=2,
                                ae_target_mode='normalized_delta',normalizers=norm,device='cpu')
                            error = float((reconstructed-expected).abs().max())
                            max_error = max(max_error,error)
                            torch.testing.assert_close(reconstructed,expected,rtol=0,atol=1e-6)
                    cache[split].append(record)
                    print(split,spec['name'],index,flush=True)
    assert digest(ae.state_dict())==before
    assert sha(checkpoint)==provenance['ae_sha256']
    provenance.update(frozen_ae_digest=before,decoder_parity_max_abs=max_error,
                      train_count=len(cache['train']),validation_count=len(cache['validation']),seconds=time.time()-start)
    torch.save(cache,out)
    out.with_suffix('.completed.json').write_text(json.dumps(dict(status='completed',cache_sha256=sha(out),
        **provenance),indent=2)+'\n')


if __name__=='__main__':
    main()
