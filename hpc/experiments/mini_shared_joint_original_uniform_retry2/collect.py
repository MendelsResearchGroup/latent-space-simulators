"""Post-fit physical evaluation and latent coordinates; never used for selection."""
import argparse,copy,gc,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lss.dynamics.capacity import load_experiment_bundle
from lss.dynamics.experiment import resolve_train_val_test
from lss.dynamics.training import encode_frame_latent
from graph_utils import directional_side_indices_from_box
from evaluate import evaluate,physical,_pref,_p_ratio
from train import OUT,CODE,DATA_OUT,sha,dump

@torch.no_grad()
def latents(result, sims):
    rows=[];model=result['ae'];model.eval()
    for sim in sims:
        reference=sim[0];ref,_=_pref(reference);sides=directional_side_indices_from_box(ref,quantile=.10)
        refxy=ref.x[:,:2].numpy()
        def widths(x):
            return np.array([x[sides['right'],0].mean()-x[sides['left'],0].mean(),x[sides['top'],1].mean()-x[sides['bottom'],1].mean()])
        initial=widths(refxy)
        for frame in range(200):
            z=encode_frame_latent(model,sim,frame,pos_dim=2,node_feature_mode=result['params']['node_feature_mode'],normalizers=result['normalizers'],device='cpu').detach().cpu().reshape(-1)
            assert len(z)==2
            graph=physical(sim[frame],reference)
            if frame==0: graph.x[:,:2]=ref.x[:,:2]
            deformation=widths(graph.x[:,:2].numpy())-initial
            rows.append(dict(source=str(reference.source_name),trajectory_id=int(reference.source_sim_id),frame=frame,
                z0=float(z[0]),z1=float(z[1]),p_ratio=_p_ratio(ref,graph,sides),x_deformation=float(deformation[0]),y_deformation=float(deformation[1])))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    folder=OUT/(a.arm+('_smoke' if a.smoke else ''))
    assert json.loads((folder/'fit_completed.json').read_text())['status']=='completed'
    recipe=json.loads((folder/'propagator_recipe.json').read_text())
    result=load_experiment_bundle(folder/'propagator.pt',cfg=recipe['config'],device='cpu')
    if a.smoke:
        evaluate({**result,'val_data':result['val_data'][:1]},seed=123,latent_dim=2,output=folder/'validation',horizon=199,split='validation')
        latents(result,result['val_data'][:1]).to_csv(folder/'validation/latents.csv',index=False)
        dump(folder/'completed.json',dict(status='completed',smoke=True))
        return
    if a.arm.startswith('single_'):
        sims=result['train_data'][:1]
        table=evaluate({**result,'val_data':sims},seed=123,latent_dim=2,output=folder/'demo',horizon=199,split='validation')
        table['split']='training_demo';table.to_csv(folder/'demo/per_network.csv',index=False)
        # Per-frame R² over a single trajectory is undefined; notebook shows response traces.
        (folder/'demo/per_step.csv').unlink()
        latents(result,sims).to_csv(folder/'demo/latents.csv',index=False)
        manifest=json.loads((DATA_OUT/'data_manifest.json').read_text())
        category={'single_node':'node','single_stiffness':'stiffness'}[a.arm]
        # Same network at a new temperature is distinct from unseen-network transfer.
        for name, cache, indices, offset, destination, label in [
            (category+'_mixed','trainval',[0],0,'paired_transfer','paired_temperature_transfer'),
            (category+'_original','test',list(range(20)),40,'unseen_transfer/original','unseen_original'),
            (category+'_mixed','test',list(range(20)),40,'unseen_transfer/mixed','unseen_mixed_temperature'),
        ]:
            entry=manifest['datasets'][name]
            assert sha(entry[cache+'_path'])==entry[cache+'_sha256']
            spec=dict(name=name,label=name,path=entry[cache+'_path'],train_count=0,val_count=len(indices),
                      split_indices=dict(train=[],val=indices,test=[]))
            source=dict(name=name,label=name,path=entry[cache+'_path'],dataset_name=name,source_name=name,
                        coordinate_normalization='position_normalization',dataset_mixture=[spec])
            _,transfer,_,_=resolve_train_val_test(source,result['params'],split_seed=123)
            for index,sim in zip(indices,transfer):
                for frame in sim: frame.source_sim_id=index+offset
            rows=evaluate({**result,'val_data':transfer},seed=123,latent_dim=2,output=folder/destination,horizon=199,split='test')
            rows['split']=label;rows.to_csv(folder/destination/'per_network.csv',index=False)
            latents(result,transfer).to_csv(folder/destination/'latents.csv',index=False)
            if len(indices)==1: (folder/destination/'per_step.csv').unlink()
            del transfer;gc.collect()
    else:
        manifest=json.loads((DATA_OUT/'data_manifest.json').read_text())
        all_steps=[];all_rows=[];all_latents=[]
        for name,entry in manifest['datasets'].items():
            assert sha(entry['test_path'])==entry['test_sha256']
            spec=dict(name=name,label=name,path=entry['test_path'],train_count=0,val_count=80,split_indices=dict(train=[],val=list(range(80)),test=[]))
            source=dict(name=name,label=name,path=entry['test_path'],dataset_name=name,source_name=name,coordinate_normalization='position_normalization',dataset_mixture=[spec])
            _,sims,_,_=resolve_train_val_test(source,result['params'],split_seed=123)
            assert len(sims)==80
            # Cached test indices0..79 correspond exactly to original cohort indices40..119.
            for i,sim in enumerate(sims):
                for frame in sim: frame.source_sim_id=i+40
            rows=evaluate({**result,'val_data':sims},seed=123,latent_dim=2,output=folder/'test'/name,horizon=199,split='test')
            all_rows.append(rows);all_steps.append(pd.read_csv(folder/'test'/name/'per_step.csv'));all_latents.append(latents(result,sims))
            del sims;gc.collect()
        pd.concat(all_rows,ignore_index=True).to_csv(folder/'test/per_network.csv',index=False)
        pd.concat(all_steps,ignore_index=True).to_csv(folder/'test/per_step.csv',index=False)
        pd.concat(all_latents,ignore_index=True).to_csv(folder/'test/latents.csv',index=False)
    dump(folder/'completed.json',dict(status='completed',job_id=os.environ['PBS_JOBID'],ae_sha256=sha(folder/'ae.pt'),propagator_sha256=sha(folder/'propagator.pt'),test_used_for_selection=False))

if __name__=='__main__': main()
