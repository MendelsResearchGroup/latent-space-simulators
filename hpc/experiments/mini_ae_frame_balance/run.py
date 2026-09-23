"""Matched short AE fine-tuning: uniform versus training-motion-balanced frames."""
import argparse,copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lss.dynamics.experiment import _load_ae_cache
from lss.dynamics.simulation import batch_delta_graphs,ae_target_tensor
from evaluate import evaluate_autoencoder
ROOT=Path(os.environ['LSS_PROJECT_ROOT']);CODE=Path(__file__).resolve().parent
OUT=ROOT/'notebooks/results/mini_ae_frame_balance'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(b''.join(k.encode()+v.cpu().numpy().tobytes() for k,v in sorted(d.items()))).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--study',choices=['mini_node','mini_stiffness'],required=True);p.add_argument('--mode',choices=['uniform','balanced'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']));torch.manual_seed(123);np.random.seed(123);random.seed(123)
base=ROOT/'notebooks/results'/a.study/'runs/d2_s123';original=json.loads((base/'recipe.json').read_text())
result=_load_ae_cache(base/'ae.pt',original['config'],device='cpu');model=result['ae'];norm=result['normalizers'];params=result['params']
assert len(result['train_data'])==len(result['val_data'])==20
out=OUT/f'{a.study}_{a.mode}{"_smoke" if a.smoke else ""}';out.mkdir(parents=True,exist_ok=False)
start=time.time();original_hash=sha(base/'ae.pt');stats_hash=digest(norm)
for parameter in model.parameters():parameter.requires_grad_(True)
def batch(sims,rows):return batch_delta_graphs(sims,rows,pos_dim=2,device='cpu',node_feature_mode=params['node_feature_mode'],edge_mode=params['edge_mode'])
# Estimate each frame's normalized displacement energy from training networks only.
energy=[]
for frame in range(20):
    data=batch(result['train_data'],[(i,frame) for i in range(20)])
    target=ae_target_tensor(data,params['ae_target_mode'])/norm['target_std']
    energy.append(torch.stack([target[data['batch']==i].square().mean() for i in range(20)]).mean())
energy=torch.stack(energy).detach();floor=.04*energy[1:].median()
assert floor>0
balanced=energy.clamp_min(floor).reciprocal();balanced/=balanced.mean()
weights=balanced if a.mode=='balanced' else torch.ones(20)
recipe=dict(study=a.study,mode=a.mode,seed=123,smoke=a.smoke,source=original['source'],starting_ae_sha256=original_hash,
    normalizers_sha256=stats_hash,frame_energy=energy.tolist(),energy_floor=float(floor),frame_weights=weights.tolist(),
    objective='Mean per-graph target-normalized displacement MSE times training-only frame weight',
    selection='Same weighted validation displacement objective; epoch0 eligible; min_delta1e-8',
    optimizer=dict(name='Adam',lr=1e-4,weight_decay=1e-5,clip_norm=1,max_epochs=30,patience=8),
    final_test_used=False,response_selection=False,job_id=os.environ['PBS_JOBID'],code_sha256=sha(CODE/'manifest.json'))
(out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
def loss(sims,rows):
    b=batch(sims,rows)
    target=(ae_target_tensor(b,params['ae_target_mode'])-norm['target_mean'])/norm['target_std']
    prediction,_=model((b['node_feature']-norm['node_feature_mean'])/norm['node_feature_std'],b['ref_pos'],
        (b['edge_attr']-norm['edge_mean'])/norm['edge_std'],
        (b['ref_edge_attr']-norm['ref_edge_mean'])/norm['ref_edge_std'],b['edge_index'],b['batch'])
    pernode=(prediction-target).square().mean(-1)
    pergraph=torch.stack([pernode[b['batch']==i].mean() for i in range(len(rows))])
    return (pergraph*weights[torch.tensor([r[1] for r in rows])]).mean()
count=2 if a.smoke else 20
rows=[(i,f) for i in range(count) for f in range(20)]
@torch.no_grad()
def validation():
    model.eval();total=0.
    for start in range(0,len(rows),32):
        group=rows[start:start+32];total+=float(loss(result['val_data'],group))*len(group)
    return total/len(rows)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
history=[dict(epoch=0,val_loss=validation())];best=history[0]['val_loss'];best_epoch=0;state=copy.deepcopy(model.state_dict());bad=0
for epoch in range(1,2 if a.smoke else 31):
    order=rows.copy();random.shuffle(order);model.train();train=[]
    for offset in range(0,len(order),32):
        objective=loss(result['train_data'],order[offset:offset+32]);optimizer.zero_grad();objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();train.append(float(objective.detach()))
    value=validation();history.append(dict(epoch=epoch,train_loss=float(np.mean(train)),val_loss=value));pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    if value<best-1e-8:best=value;best_epoch=epoch;state=copy.deepcopy(model.state_dict());bad=0
    else:bad+=1
    print(epoch,value,flush=True)
    if bad>=8:break
model.load_state_dict(state);model.eval()
assert digest(norm)==stats_hash and sha(base/'ae.pt')==original_hash
bundle=torch.load(base/'ae.pt',map_location='cpu',weights_only=False);bundle['ae_state_dict']=state;bundle['frame_balance_recipe']=recipe;torch.save(bundle,out/'ae.pt')
evaluate_autoencoder({**result,'val_data':result['val_data'][:count]},seed=123,latent_dim=2,output=out/'validation',horizon=19,split='validation')
(out/'completed.json').write_text(json.dumps(dict(status='completed',job_id=os.environ['PBS_JOBID'],best_epoch=best_epoch,epochs=len(history)-1,seconds=time.time()-start,ae_sha256=sha(out/'ae.pt'),normalizers_unchanged=True,original_ae_unchanged=True),indent=2)+'\n')
