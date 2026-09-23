#!/usr/bin/env python3
"""Train the dense Reid prefix AE, then frozen-AE latent dynamics."""
from pathlib import Path
import argparse
import gc
import copy
import hashlib
import json
import os
import resource
import time
import torch
from lss.dynamics.experiment import run_latent_experiment, seed_everything, resolve_train_val_test
from evaluate import evaluate, evaluate_autoencoder

ROOT = Path(os.environ['LSS_PROJECT_ROOT'])
STUDY = ROOT / 'notebooks/results/dense_reid'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, default=str)+'\n')

def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def config(dimension, seed, out, smoke):
    return {
        'ae_config': {
            'model':'message_passing_reference16', 'latent_dim':dimension,
            'latent_tokens':32, 'hidden_size':96, 'edge_feature_dim':4,
            'message_passing_steps':2, 'node_feature_mode':'normalized_delta',
            'target_mode':'normalized_delta',
            'max_train_frames_per_sim':101, 'max_val_frames_per_sim':101,
            'max_epochs':1 if smoke else 60, 'patience':1 if smoke else 12,
            'lr':1e-4, 'weight_decay':1e-5, 'mix_sources':True,
            'balance_sources':False, 'gradient_method':'source_mean',
            'checkpoint_metric':'val_max_source_reconstruction', 'checkpoint_mode':'min',
            'pratio_eval_every':0,
        },
        'coordinate_normalization':'position_normalization', 'edge_mode':'compact_stored',
        'static_context_use_physical_reference':True, 'pos_dim':2,
        'batch_graphs':32, 'frame_skip':1, 'early_stop_min_delta':1e-5,
        'split_seed':123, 'model_seed':seed, 'should_train_propagator':False,
        'should_rollout':False, 'force_train':True, 'cache_require_matching_config':True,
        'cache_path':str(out/'ae.pt'),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dimension',type=int,choices=(2,4),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); start=time.time()
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    seed_everything(a.seed)
    out=STUDY/'runs'/f'd{a.dimension}_s{a.seed}{"_smoke" if a.smoke else ""}'
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((STUDY/'prepared.json').read_text())
    source=manifest['source']
    if a.smoke:
        source=json.loads(json.dumps(source))
        spec=source['dataset_mixture'][0]
        spec['split_indices']['train']=spec['split_indices']['train'][:2]
        spec['split_indices']['val']=spec['split_indices']['val'][:2]
        spec['train_count']=2; spec['val_count']=2
    path=Path(source['path'])
    assert sha(path)==manifest['prefix_sha256']
    code=Path(os.environ['LSS_CODE_ROOT'])
    hashes=json.loads((code/'hashes.json').read_text())
    assert all(sha(code/k)==v for k,v in hashes.items())
    cfg=config(a.dimension,a.seed,out,a.smoke)
    recipe={'source':source,'config':cfg,'seed':a.seed,'dimension':a.dimension,
            'data_manifest_sha256':sha(STUDY/'prepared.json'),
            'code_manifest_sha256':sha(code/'hashes.json'),'job_id':os.environ.get('PBS_JOBID','local'),
            'horizon':100,'frame_indices':list(range(101)), 'final_test_used_for_selection':False,
            'final_test_policy':'Evaluate all frozen predeclared models after fitting; smoke excludes test',
            'response_selection':False,'selection':'AE validation displacement loss; propagator validation latent delta loss',
            'strategies':['one_step','multistep8']}
    dump(out/'recipe.json',recipe)
    ae_result=run_latent_experiment(source,cfg,device='cpu')
    ae_result['ae_history'].to_csv(out/'ae_history.csv',index=False)
    expected=tensor_hash(ae_result['ae'].state_dict())
    normalizer_keys=('target_mean','target_std','node_feature_mean','node_feature_std',
                     'edge_mean','edge_std','ref_edge_mean','ref_edge_std')
    frozen_normalizers={k:ae_result['normalizers'][k].detach().cpu().clone() for k in normalizer_keys}
    ae_sha=sha(out/'ae.pt')
    dump(out/'ae_completed.json',{'sha256':ae_sha,'state_sha256':expected,'elapsed_seconds':time.time()-start})
    evaluate_autoencoder(ae_result,seed=a.seed,latent_dim=a.dimension,
        output=out/'ae_evaluation'/'validation',horizon=100,split='validation')
    del ae_result; gc.collect()
    for strategy in recipe['strategies']:
        seed_everything(a.seed)
        run=out/strategy; run.mkdir()
        dyn_cfg={**cfg,
            'should_train_propagator':True, 'pretrained_ae_cache_path':str(out/'ae.pt'),
            'pretrained_ae_config_keys':['autoencoder_model','latent_dim','latent_tokens','hidden_size',
                'edge_feature_dim','message_passing_steps','node_feature_mode','ae_target_mode','edge_mode','pos_dim'],
            'pretrained_ae_require_matching_config':True,'pretrained_ae_skip_stat_fitting':True,
            'pretrained_ae_require_matching_normalizers':False,
            'cache_path':str(run/'propagator.pt'), 'propagator_model':'delta_mlp',
            'propagator_loss':'delta','propagator_objective':'one_step' if strategy=='one_step' else 'multistep',
            'propagator_multistep_horizons':list(range(1,9)), 'propagator_hidden_size':64,
            'propagator_use_static_context':True,'propagator_context_pool':'mean','graph_context_dim':16,
            'propagator_context_include_temperature':False,'propagator_context_include_source_id':False,
            'propagator_source_loss_reduction':'equal','propagator_position_loss_weight':0.,
            'physics_loss_enabled':False,'propagator_checkpoint_metric':'val_loss',
            'propagator_checkpoint_mode':'min','propagator_rollout_eval_every_epoch':False,
            'dyn_max_epochs':1 if a.smoke else 30,'dyn_patience':1 if a.smoke else 6,
            'dyn_lr':1e-4,'dyn_weight_decay':1e-5,'dyn_max_train_transitions_per_sim':100,
            'dyn_max_val_transitions_per_sim':100,'rollout_steps_grid':[100],'rollout_eval_splits':[],
        }
        dump(run/'recipe.json',{'config':dyn_cfg,'ae_sha256':ae_sha,'strategy':strategy})
        result=run_latent_experiment(source,dyn_cfg,device='cpu')
        assert tensor_hash(result['ae'].state_dict())==expected
        assert sha(out/'ae.pt')==ae_sha
        assert all(torch.equal(result['normalizers'][k].detach().cpu(),v) for k,v in frozen_normalizers.items())
        result['dyn_history'].to_csv(run/'dynamics_history.csv',index=False)
        evaluate(result,seed=a.seed,latent_dim=a.dimension,output=run/'validation',horizon=100,split='validation')
        if not a.smoke:
            test_source=copy.deepcopy(source)
            spec=test_source['dataset_mixture'][0]
            spec['train_count']=0; spec['val_count']=len(manifest['test_indices'])
            spec['split_indices']={'train':[],'val':manifest['test_indices'],'test':[]}
            _, test_data, _, _=resolve_train_val_test(test_source,result['params'],split_seed=123)
            evaluate({**result,'val_data':test_data},seed=a.seed,latent_dim=a.dimension,
                output=run/'test',horizon=100,split='test')
            del test_data
        dump(run/'completed.json',{'status':'completed','ae_unchanged':True,'bundle_sha256':sha(run/'propagator.pt')})
        del result; gc.collect()
    dump(out/'completed.json',{'status':'completed','job_id':os.environ.get('PBS_JOBID','local'),
        'seconds':time.time()-start,'max_rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

if __name__=='__main__':
    main()
