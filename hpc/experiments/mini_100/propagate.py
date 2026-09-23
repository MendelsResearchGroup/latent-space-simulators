"""Fit the one-step latent propagator after an AE is frozen, then evaluate all test frames."""
import argparse,copy,gc,hashlib,json,os
from pathlib import Path
import torch
from lss.dynamics.experiment import run_latent_experiment,resolve_train_val_test,seed_everything
from lss.dynamics.capacity import load_experiment_bundle
from evaluate import evaluate
ROOT=Path(os.environ['LSS_PROJECT_ROOT']);OUT=ROOT/'notebooks/results/mini_100'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
folder=OUT/f'{a.study}_{a.mode}{"_smoke" if a.smoke else ""}'
assert json.loads((folder/'completed.json').read_text())['status']=='completed'
prepared=json.loads((OUT/f'{a.study}_prepared.json').read_text());source=copy.deepcopy(prepared['source'])
if a.smoke:
    spec=source['dataset_mixture'][0]
    spec['split_indices']['train']=spec['split_indices']['train'][:2];spec['split_indices']['val']=spec['split_indices']['val'][:2]
    spec['train_count']=spec['val_count']=2
cfg=json.loads((ROOT/'notebooks/results'/a.study/'runs/d2_s123/recipe.json').read_text())['config']
cfg['ae_config']['max_train_frames_per_sim']=100;cfg['ae_config']['max_val_frames_per_sim']=100
cfg.update(should_train_propagator=True,should_rollout=False,pretrained_ae_cache_path=str(folder/'ae.pt'),
    pretrained_ae_config_keys=['autoencoder_model','latent_dim','latent_tokens','hidden_size','edge_feature_dim','message_passing_steps','node_feature_mode','ae_target_mode','edge_mode','pos_dim'],
    pretrained_ae_require_matching_config=True,pretrained_ae_skip_stat_fitting=True,pretrained_ae_require_matching_normalizers=False,
    propagator_model='delta_mlp',propagator_loss='delta',
    propagator_hidden_size=64,propagator_use_static_context=True,propagator_context_pool='mean',graph_context_dim=16,
    propagator_context_include_temperature=False,propagator_context_include_source_id=False,
    propagator_source_loss_reduction='equal',propagator_position_loss_weight=0.,physics_loss_enabled=False,
    propagator_checkpoint_metric='val_loss',propagator_checkpoint_mode='min',propagator_rollout_eval_every_epoch=False,
    dyn_max_epochs=1 if a.smoke else 60,dyn_patience=1 if a.smoke else 12,dyn_lr=1e-4,dyn_weight_decay=1e-5,
    dyn_max_train_transitions_per_sim=99,dyn_max_val_transitions_per_sim=99,rollout_steps_grid=[99],rollout_eval_splits=[])
ae_sha=sha(folder/'ae.pt');configs={}
for strategy in ('one_step',):
    seed_everything(123);run=folder/strategy;run.mkdir()
    current=copy.deepcopy(cfg);current.update(cache_path=str(run/'propagator.pt'),propagator_objective='one_step')
    configs[strategy]=current
    (run/'recipe.json').write_text(json.dumps(dict(config=current,source=source,ae_sha256=ae_sha,job_id=os.environ['PBS_JOBID'],test_used_for_selection=False),indent=2)+'\n')
    result=run_latent_experiment(source,current,device='cpu')
    saved=torch.load(folder/'ae.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),saved['ae_state_dict'][k].cpu()) for k,v in result['ae'].state_dict().items())
    assert sha(folder/'ae.pt')==ae_sha
    result['dyn_history'].to_csv(run/'history.csv',index=False)
    del result,saved;gc.collect()
# Neither test response nor test positions are consulted until the fit finishes.
for strategy,current in configs.items():
    run=folder/strategy;result=load_experiment_bundle(run/'propagator.pt',cfg=current,device='cpu')
    evaluate(result,seed=123,latent_dim=2,output=run/'validation',horizon=99,split='validation')
    if not a.smoke:
        evaluation=copy.deepcopy(source);spec=evaluation['dataset_mixture'][0]
        spec['train_count']=0;spec['val_count']=80;spec['split_indices']={'train':[],'val':prepared['test_indices'],'test':[]}
        _,test_data,_,_=resolve_train_val_test(evaluation,result['params'],split_seed=123)
        evaluate({**result,'val_data':test_data},seed=123,latent_dim=2,output=run/'test',horizon=99,split='test')
        del test_data
    (run/'completed.json').write_text(json.dumps(dict(status='completed',ae_unchanged=True,checkpoint_sha256=sha(run/'propagator.pt')))+'\n')
    del result;gc.collect()
(folder/'pipeline_completed.json').write_text(json.dumps(dict(status='completed',job_id=os.environ['PBS_JOBID']))+'\n')
