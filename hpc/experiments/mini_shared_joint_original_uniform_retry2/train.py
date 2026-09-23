"""Matched 2D reference-message-passing AE and frozen one-step latent dynamics."""
import argparse, copy, gc, hashlib, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lss.dynamics.experiment import run_latent_experiment, seed_everything
from lss.dynamics.simulation import batch_delta_graphs, ae_target_tensor

ROOT=Path(os.environ['LSS_PROJECT_ROOT'])
OUT=ROOT/'notebooks/results/mini_shared'
DATA_OUT=ROOT/'notebooks/results/mini_shared'
CODE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def dump(path,value): path.write_text(json.dumps(value,indent=2)+'\n')

def source_spec(names, manifest, single, smoke):
    specs=[]
    for name in names:
        n=1 if single or smoke else 20
        specs.append(dict(name=name,label=name,path=manifest['datasets'][name]['trainval_path'],train_count=n,
            val_count=1 if smoke else 20,split_indices=dict(train=list(range(n)),val=list(range(20,21 if smoke else 40)),test=[])))
    return dict(name='mini_shared',dataset_name='mini_shared',source_name='mini_shared',label='mini_shared',
        path=specs[0]['path'],dataset_mixture=specs,coordinate_normalization='position_normalization')

def config(out):
    return dict(ae_config=dict(model='message_passing_reference16',latent_dim=2,latent_tokens=32,hidden_size=96,
        edge_feature_dim=4,message_passing_steps=2,node_feature_mode='normalized_delta',target_mode='normalized_delta',
        max_train_frames_per_sim=200,max_val_frames_per_sim=200,max_epochs=1,patience=1,lr=1e-4,weight_decay=1e-5,
        mix_sources=True,balance_sources=False,gradient_method='source_mean',checkpoint_metric='val_max_source_reconstruction',
        checkpoint_mode='min',pratio_eval_every=0),coordinate_normalization='position_normalization',edge_mode='compact_stored',
        static_context_use_physical_reference=True,pos_dim=2,batch_graphs=32,frame_skip=1,early_stop_min_delta=1e-8,
        split_seed=123,model_seed=123,should_train_propagator=False,should_rollout=False,force_train=True,
        cache_require_matching_config=True,cache_path=str(out/'ae.pt'))

def assert_historical_config(cfg, out):
    baseline=DATA_OUT/'joint_original'
    expected=json.loads((baseline/'recipe.json').read_text())['config']
    expected['cache_path']=str(out/'ae.pt')
    assert cfg==expected, 'Uniform arm changed an AE configuration beyond its output path.'

def assert_historical_propagator_config(dyn_cfg, out, smoke):
    expected=json.loads((DATA_OUT/'joint_original'/'propagator_recipe.json').read_text())['config']
    expected['cache_path']=str(out/'propagator.pt');expected['pretrained_ae_cache_path']=str(out/'ae.pt')
    if smoke:
        expected['dyn_max_epochs']=1;expected['dyn_patience']=1
    assert dyn_cfg==expected, 'Uniform arm changed a propagator configuration beyond output paths and smoke-only budget.'

def propagator_config(cfg, out, smoke):
    dyn_cfg=copy.deepcopy(cfg);dyn_cfg.update(should_train_propagator=True,pretrained_ae_cache_path=str(out/'ae.pt'),
        pretrained_ae_config_keys=['autoencoder_model','latent_dim','latent_tokens','hidden_size','edge_feature_dim','message_passing_steps','node_feature_mode','ae_target_mode','edge_mode','pos_dim'],
        pretrained_ae_require_matching_config=True,pretrained_ae_skip_stat_fitting=True,pretrained_ae_require_matching_normalizers=False,
        cache_path=str(out/'propagator.pt'),propagator_model='delta_mlp',propagator_loss='delta',propagator_objective='one_step',
        propagator_hidden_size=64,propagator_use_static_context=True,propagator_context_pool='mean',graph_context_dim=16,
        propagator_context_include_temperature=False,propagator_context_include_source_id=False,propagator_source_loss_reduction='equal',
        propagator_position_loss_weight=0.,physics_loss_enabled=False,propagator_checkpoint_metric='val_loss',propagator_checkpoint_mode='min',
        propagator_rollout_eval_every_epoch=False,dyn_max_epochs=1 if smoke else 60,dyn_patience=1 if smoke else 12,
        dyn_lr=1e-4,dyn_weight_decay=1e-5,dyn_max_train_transitions_per_sim=199,dyn_max_val_transitions_per_sim=199,
        rollout_steps_grid=[199],rollout_eval_splits=[])
    assert_historical_propagator_config(dyn_cfg,out,smoke)
    return dyn_cfg

def fit_uniform(result, out, epochs):
    model,norm,params=result['ae'],result['normalizers'],result['params']
    for p in model.parameters(): p.requires_grad_(True)
    train,val=result['train_data'],result['val_data']
    names=sorted({str(s[0].source_name) for s in train})
    def batch(sims,rows):
        return batch_delta_graphs(sims,rows,pos_dim=2,device='cpu',node_feature_mode=params['node_feature_mode'],edge_mode=params['edge_mode'])
    energy={}
    with torch.no_grad():
        for name in names:
            ids=[i for i,s in enumerate(train) if str(s[0].source_name)==name]
            values=[]
            for frame in range(200):
                b=batch(train,[(i,frame) for i in ids]); y=ae_target_tensor(b,params['ae_target_mode'])/norm['target_std']
                values.append(torch.stack([y[b['batch']==j].square().mean() for j in range(len(ids))]).mean())
            energy[name]=torch.stack(values)
    weights={name:torch.ones_like(value) for name,value in energy.items()}
    assert all(bool(torch.equal(weight,torch.ones_like(weight))) for weight in weights.values())
    dump(out/'frame_weights.json',{n:dict(energy_diagnostic=energy[n].tolist(),weights=weights[n].tolist(),
        post_warmup_weighting='uniform_exactly_1_per_frame') for n in names})
    def objective(sims,rows):
        b=batch(sims,rows)
        target=(ae_target_tensor(b,params['ae_target_mode'])-norm['target_mean'])/norm['target_std']
        prediction,_=model((b['node_feature']-norm['node_feature_mean'])/norm['node_feature_std'],b['ref_pos'],
            (b['edge_attr']-norm['edge_mean'])/norm['edge_std'],(b['ref_edge_attr']-norm['ref_edge_mean'])/norm['ref_edge_std'],b['edge_index'],b['batch'])
        error=(prediction-target).square().mean(-1)
        return torch.stack([error[b['batch']==j].mean()*weights[str(sims[i][0].source_name)][f] for j,(i,f) in enumerate(rows)]).mean()
    train_rows=[(i,f) for i in range(len(train)) for f in range(200)]
    @torch.no_grad()
    def validation():
        model.eval(); scores=[]
        for name in names:
            rows=[(i,f) for i,s in enumerate(val) if str(s[0].source_name)==name for f in range(200)]
            total=0.
            for j in range(0,len(rows),32): total+=float(objective(val,rows[j:j+32]))*len(rows[j:j+32])
            scores.append(total/len(rows))
        return float(np.mean(scores))
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    best=validation();state=copy.deepcopy(model.state_dict());bad=0;history=[dict(epoch=0,val_loss=best)]
    for epoch in range(1,epochs+1):
        random.shuffle(train_rows);model.train();total=0.
        for j in range(0,len(train_rows),32):
            rows=train_rows[j:j+32];loss=objective(train,rows);opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(rows)
        score=validation();history.append(dict(epoch=epoch,train_loss=total/len(train_rows),val_loss=score))
        pd.DataFrame(history).to_csv(out/'ae_history.csv',index=False)
        print('uniform AE',epoch,score,flush=True)
        if score<best-1e-8: best=score;state=copy.deepcopy(model.state_dict());bad=0
        else: bad+=1
        if bad>=12: break
    model.load_state_dict(state)
    bundle=torch.load(out/'ae.pt',map_location='cpu',weights_only=False)
    bundle['ae_state_dict']=state;bundle['ae_history']=history
    bundle['frame_balance_recipe']=dict(training_only=True,post_warmup_weighting='uniform_exactly_1_per_frame',warmup_epochs=1)
    torch.save(bundle,out/'ae.pt')
    return sha(out/'ae.pt')

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--smoke',action='store_true');p.add_argument('--assert-config',action='store_true');a=p.parse_args()
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']));seed_everything(123)
    if a.assert_config:
        for smoke in (True,False):
            out=OUT/('static_smoke' if smoke else 'static_production');cfg=config(out);assert_historical_config(cfg,out);propagator_config(cfg,out,smoke)
        print('static AE and propagator parity assertions passed for smoke and production',flush=True);return
    if not a.arm: p.error('--arm is required unless --assert-config is used')
    protocol=json.loads((CODE/'protocol.json').read_text()); manifest=json.loads((DATA_OUT/'data_manifest.json').read_text())
    names=protocol['arms'][a.arm];single=a.arm.startswith('single_')
    out=OUT/(a.arm+('_smoke' if a.smoke else ''));out.mkdir(exist_ok=False)
    source=source_spec(names,manifest,single,a.smoke);cfg=config(out);assert_historical_config(cfg,out)
    assert tuple(names)==('stiffness_original','node_original')
    assert not any('mixed' in name for name in names)
    for n in names: assert sha(manifest['datasets'][n]['trainval_path'])==manifest['datasets'][n]['trainval_sha256']
    dump(out/'recipe.json',dict(config=cfg,source=source,arm=a.arm,seed=123,job_id=os.environ['PBS_JOBID'],
        protocol_sha256=sha(CODE/'protocol.json'),code_sha256=sha(CODE/'manifest.json'),data_manifest_sha256=sha(CODE/'data_manifest.json'),
        ae_objective='one ordinary warmup epoch, then uniform exactly-1-per-frame/source displacement MSE',
        selection='uniform validation displacement MSE; propagator validation latent delta MSE',test_selection=False,
        mixed_temperature_excluded_from_fitting_normalization_and_selection=True))
    result=run_latent_experiment(source,cfg,device='cpu')
    result['ae_history'].to_csv(out/'ae_warmup_history.csv',index=False)
    ae_sha=fit_uniform(result,out,1 if a.smoke else 60)
    del result;gc.collect()
    dump(out/'ae_completed.json',dict(status='completed',sha256=ae_sha))
    dyn_cfg=propagator_config(cfg,out,a.smoke)
    dump(out/'propagator_recipe.json',dict(config=dyn_cfg,ae_sha256=ae_sha))
    seed_everything(123);result=run_latent_experiment(source,dyn_cfg,device='cpu')
    frozen=torch.load(out/'ae.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v.cpu(),frozen['ae_state_dict'][k].cpu()) for k,v in result['ae'].state_dict().items())
    assert sha(out/'ae.pt')==ae_sha
    result['dyn_history'].to_csv(out/'propagator_history.csv',index=False)
    dump(out/'fit_completed.json',dict(status='completed',ae_unchanged=True,propagator_sha256=sha(out/'propagator.pt')))

if __name__=='__main__': main()
