"""Matched node-reference widths and paired context/no-context dynamics."""
import argparse,copy,gc,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
ROOT=Path(os.environ['LSS_PROJECT_ROOT']);CODE=Path(os.environ['LSS_CODE_ROOT'])
RESULTS_ROOT=Path(os.environ.get('LSS_RESULTS_ROOT',ROOT/'notebooks/results'))
def load_portable_manifest(path):
    manifest=json.loads(path.read_text())
    for spec in manifest.get('sources',{}).values():spec['path']=str(ROOT/'data'/Path(spec['path']).name)
    return manifest
sys.path[:0]=[str(CODE/'src')]
import numpy as np
import pandas as pd
import torch
from lss.dynamics.experiment import (run_latent_experiment,seed_everything,resolve_train_val_test,
    evaluate_rollout_horizons,evaluate_autoencoder_reconstruction_horizons,latent_experiment_cache_key)
from lss.dynamics.capacity import load_experiment_bundle

SOURCES=('reid','depablo_low_temp')
MODELS={96:'attention',16:'attention_reference16',8:'attention_reference8'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()

def weights(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=str))

def config(args,out,manifest):
    n=2 if args.smoke else 20
    specs=[]
    for source in SOURCES:
        m=manifest['sources'][source]
        specs.append(dict(name=source,label=source,path=m['path'],train_count=n,val_count=n,
            split_indices=dict(train=m['split_indices']['train'][:n],val=m['split_indices']['val'][:n],test=[])))
    source=dict(dataset_name='network_design_reid_lowT',source_name='reid_lowT',label='Reid + low-T 2D AE',path=specs[0]['path'],dataset_mixture=specs)
    cfg=dict(ae_config=dict(model=MODELS[args.reference_dim],latent_dim=args.latent_dim,latent_tokens=32,hidden_size=96,
        node_feature_mode='normalized_delta',target_mode='normalized_delta',edge_feature_dim=4,
        max_train_frames_per_sim=4 if args.smoke else 200,max_val_frames_per_sim=4 if args.smoke else 200,
        max_epochs=1 if args.smoke else 40,patience=1 if args.smoke else 8,
        lr=1e-4,weight_decay=1e-5,mix_sources=True,balance_sources=False,gradient_method='source_mean',
        checkpoint_metric='val_max_source_reconstruction',checkpoint_mode='min',pratio_eval_every=0),
        coordinate_normalization='position_normalization',edge_mode='compact_stored',
        static_context_use_physical_reference=True,pos_dim=2,batch_graphs=32,frame_skip=1,
        early_stop_min_delta=1e-5,split_seed=123,model_seed=args.seed,should_rollout=False,should_train_propagator=False,
        cache_require_matching_config=True,cache_path=str(out/'ae.pt'))
    return source,cfg


def evaluate(result,sims,ids,source,kind):
    horizon=min(len(s) for s in sims)-1
    common=dict(cfg=result['params'],normalizers=result['normalizers'],dataset=source,split_name='val',rollout_steps=[horizon],device='cpu')
    if kind=='ae':rows,_=evaluate_autoencoder_reconstruction_horizons(result['ae'],sims,**common)
    else:rows,_=evaluate_rollout_horizons(result['ae'],result['dyn'],sims,result['latent_stats'],endpoint_only=True,**common)
    rows['original_index']=rows.sim_idx.map(dict(enumerate(ids)))
    if rows.original_index.isna().any():raise ValueError('Unknown simulation identity')
    rows['source']=source;rows['kind']=kind
    rows['evaluation_role']='post-fit transfer validation' if source=='depablo_mixed_temp' else 'validation'
    return rows


def summarize(rows):
    output=[]
    for (kind,source),g in rows.groupby(['kind','source']):
        a='endpoint_true_p_ratio' if 'endpoint_true_p_ratio' in g else 'true_p_ratio'
        b='endpoint_pred_p_ratio' if 'endpoint_pred_p_ratio' in g else 'pred_p_ratio'
        valid=np.isfinite(g[a])&np.isfinite(g[b]);y=g.loc[valid,a].to_numpy();p=g.loc[valid,b].to_numpy()
        den=np.square(y-y.mean()).sum() if len(y) else 0
        coord=np.isfinite(g.final_pos_mse)
        output.append(dict(kind=kind,source=source,total=len(g),valid=int(valid.sum()),coordinate_valid=int(coord.sum()),
            p_ratio_mae=float(np.abs(y-p).mean()) if len(y) else None,
            p_ratio_r2=float(1-np.square(y-p).sum()/den) if den>0 else None,
            final_pos_mse=float(g.loc[coord,'final_pos_mse'].mean()) if coord.any() else None,
            split='validation',transfer_only=source=='depablo_mixed_temp'))
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reference-dim',type=int,choices=[8,16,96],required=True)
    parser.add_argument('--latent-dim',type=int,choices=[2,4],required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();torch.set_num_threads(8)
    version=os.environ['LSS_CODE_VERSION'];base=RESULTS_ROOT/'reference_simplification'
    name=f'r{args.reference_dim}_d{args.latent_dim}_s{args.seed}_{version}'+('_smoke' if args.smoke else '')
    out=base/name;out.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        manifest_path=Path(os.environ.get('LSS_SPLIT_MANIFEST',RESULTS_ROOT/'lj_ae_08_bridge/split_manifest.json'));manifest=load_portable_manifest(manifest_path)
        source,cfg=config(args,out,manifest)
        for s in (*SOURCES,'depablo_mixed_temp'):
            if sha(manifest['sources'][s]['path'])!=manifest['sources'][s]['sha256']:raise ValueError(f'Dataset changed: {s}')
        old=Path(os.environ.get('LSS_HISTORICAL_AE',RESULTS_ROOT/'network_design/reid_lowT_2d_s786/ae.pt'))
        reused=False
        if not args.smoke and args.reference_dim==96 and args.latent_dim==2 and args.seed==786 and old.exists():
            bundle=torch.load(old,weights_only=False,map_location='cpu')
            if bundle.get('cache_key')==latent_experiment_cache_key(source,cfg):
                cfg['cache_path']=str(old);reused=True
            del bundle
        recipe=dict(source=source,config=cfg,reference_dim=args.reference_dim,latent_dim=args.latent_dim,seed=args.seed,
            job_id=os.environ.get('PBS_JOBID'),code_version=version,code_root=str(CODE),
            code_manifest_sha256=sha(CODE/'hashes.json'),manifest_sha256=sha(manifest_path),
            contexts=[16,0],smoke=args.smoke,reused_engineering_ae=reused,
            selection='AE worst-source coordinate reconstruction; propagator validation latent-delta loss only',
            paired_design='Both context settings use the identical AE state; dynamic width96 and decoder tokens32 fixed',
            scope='Reid+lowT fitting; mixedT loaded only after both propagators finish; no final test or LJ claim')
        dump(out/'recipe.json',recipe);dump(out/'status.json',dict(stage='ae',status='running'))
        seed_everything(args.seed);ae_result=run_latent_experiment(source,cfg,device='cpu')
        assert not ae_result['test_data']
        ae_hash=weights(ae_result['ae']);ae_path=Path(cfg['cache_path'])
        ae_result['ae_history'].to_csv(out/'ae_history.csv',index=False)
        ae_rows=[]
        for spec in source['dataset_mixture']:
            sims=[s for s in ae_result['val_data'] if s[0].source_name==spec['name']]
            ae_rows.append(evaluate(ae_result,sims,spec['split_indices']['val'],spec['name'],'ae'))
        ae_params=sum(p.numel() for p in ae_result['ae'].parameters())
        pd.concat(ae_rows).to_csv(out/'ae_validation_rows.csv',index=False)
        del ae_result;gc.collect()
        retained=[]
        for context in [16,0]:
            stage=out/f'context{context}';stage.mkdir()
            prop=dict(model='delta_mlp',objective='one_step',loss='delta',hidden_size=64,
                max_train_transitions_per_sim=3 if args.smoke else 199,max_epochs=1 if args.smoke else 40,patience=1 if args.smoke else 8,
                lr=1e-4,weight_decay=1e-5,step_stride=1,mix_sources=True,balance_sources=False,source_loss_reduction='equal',
                train_trajectories_per_source={s:2 if args.smoke else 20 for s in SOURCES},
                val_trajectories_per_source={s:2 if args.smoke else 20 for s in SOURCES},
                use_static_context=bool(context),context_pool='mean',context_dim=context or None,
                context_include_temperature=False,context_include_source_id=False,
                rollout_eval_every_epoch=False,checkpoint_metric=None,checkpoint_mode='min',
                frozen_latent_cache_dir=str(stage/'frozen_latents'))
            pcfg=copy.deepcopy(cfg);pcfg.update(propagator_config=prop,pretrained_ae_cache_path=str(ae_path),
                pretrained_ae_require_matching_config=True,pretrained_ae_require_matching_normalizers=False,
                pretrained_ae_skip_stat_fitting=True,force_train_autoencoder=False,
                pretrained_ae_config_keys=['autoencoder_model','latent_dim','latent_tokens','hidden_size','node_feature_mode','ae_target_mode','edge_mode'],
                should_train_propagator=True,should_rollout=True,rollout_eval_splits=[],rollout_steps_grid=[199],
                p_ratio_estimator='endpoint',cache_path=str(stage/'bundle.pt'))
            dump(stage/'recipe.json',dict(source=source,config=pcfg,ae_file_sha256=sha(ae_path),ae_weight_sha256=ae_hash,
                context=context,reference_dim=args.reference_dim,seed=args.seed,job_id=os.environ.get('PBS_JOBID')))
            dump(out/'status.json',dict(stage=f'context{context}',status='running'))
            seed_everything(args.seed+10000)
            result=run_latent_experiment(source,pcfg,device='cpu')
            assert weights(result['ae'])==ae_hash,'Propagator run changed or retrained the paired AE'
            assert result['dyn'].context_dim==context
            assert result['params']['propagator_checkpoint_metric'] is None
            result['dyn_history'].to_csv(stage/'dyn_history.csv',index=False)
            rows=[]
            for spec in source['dataset_mixture']:
                sims=[s for s in result['val_data'] if s[0].source_name==spec['name']]
                rows.append(evaluate(result,sims,spec['split_indices']['val'],spec['name'],'rollout'))
            frame=pd.concat(rows);frame.to_csv(stage/'validation_rows.csv',index=False)
            dump(stage/'source_summary.json',summarize(frame))
            params=sum(p.numel() for p in result['dyn'].parameters())
            dump(stage/'fitted.json',dict(ae_weight_sha256=ae_hash,parameters=params,status='completed'))
            if args.smoke:
                loaded=load_experiment_bundle(stage/'bundle.pt',cfg=pcfg,device='cpu')
                assert weights(loaded['ae'])==ae_hash
                assert weights(loaded['dyn'])==weights(result['dyn'])
                del loaded
            retained.append((context,{k:result[k] for k in ['params','normalizers','ae','dyn','latent_stats']},frame))
            del result;gc.collect()
        # Load transfer trajectories after every checkpoint has been selected.
        m=manifest['sources']['depablo_mixed_temp'];ids=m['split_indices']['val'][:2 if args.smoke else 20]
        spec=dict(name='depablo_mixed_temp',label='mixedT post-fit only',path=m['path'],train_count=0,val_count=len(ids),
            split_indices=dict(train=[],val=ids,test=[]))
        transfer=dict(dataset_name='network_design_reid_lowT',source_name='depablo_mixed_temp',path=m['path'],dataset_mixture=[spec])
        train,mixed,test,_=resolve_train_val_test(transfer,retained[0][1]['params'],split_seed=123)
        assert not train and not test and len(mixed)==len(ids)
        ae_rows.append(evaluate(retained[0][1],mixed,ids,'depablo_mixed_temp','ae'))
        ae_frame=pd.concat(ae_rows);ae_frame.to_csv(out/'ae_all_validation_rows.csv',index=False)
        dump(out/'ae_source_summary.json',summarize(ae_frame))
        for context,result,frame in retained:
            mixed_frame=evaluate(result,mixed,ids,'depablo_mixed_temp','rollout')
            all_rows=pd.concat([frame,mixed_frame]);stage=out/f'context{context}'
            all_rows.to_csv(stage/'all_validation_rows.csv',index=False);dump(stage/'source_summary.json',summarize(all_rows))
        dump(out/'completed.json',dict(status='completed',seconds=time.time()-started,ae_parameters=ae_params,
            ae_weight_sha256=ae_hash,ae_file_sha256=sha(ae_path),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            job_id=os.environ.get('PBS_JOBID'),reference_dim=args.reference_dim,latent_dim=args.latent_dim,seed=args.seed,
            contexts=[16,0],smoke=args.smoke))
        dump(out/'status.json',dict(stage='complete',status='completed'))
    except Exception as exc:
        dump(out/'failed.json',dict(error=str(exc),traceback=traceback.format_exc(),job_id=os.environ.get('PBS_JOBID')))
        dump(out/'status.json',dict(status='failed'));raise

if __name__=='__main__':main()
