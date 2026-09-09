"""Frozen compact-LJ decoder fitted-code diagnostic; never updates AE weights."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(os.environ.get('LSS_PROJECT_ROOT',Path(__file__).resolve().parents[3])); RESULTS_ROOT=Path(os.environ.get('LSS_RESULTS_ROOT',ROOT/'notebooks/results')); HISTORICAL_RESULTS=Path(os.environ.get('LSS_HISTORICAL_RESULTS_ROOT',ROOT/'notebooks/results'))
FROZEN=HISTORICAL_RESULTS/'compact_lj_reconstruction/code_v2'
sys.path.insert(0,str(FROZEN/'src'))
import torch
from lss.data import load_dataset
from lss.dynamics.experiment import _autoencoder_class
from lss.dynamics.training import decode_latent_positions, encode_frame_latent
IDS=(178,199,42); FRAMES=(25,100,199); VARIANTS=('r16_d4','ljonly_r16_d4')
OUT=RESULTS_ROOT/'hpc_diagnostics/compact_lj_fitted_codes'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def model(path):
 b=torch.load(path,map_location='cpu',weights_only=False); p={**b['params'],**b['source_spec']}; s=b['stats']; n={k:s[k].float() for k in ('target_mean','target_std','node_feature_mean','node_feature_std','edge_mean','edge_std')}
 for k in ('ref_edge_mean','ref_edge_std'):
  if k in s:n[k]=s[k].float()
 cls=_autoencoder_class(str(p.get('autoencoder_model','attention')).lower())
 a=cls(pos_dim=int(p['pos_dim']),node_feature_dim=n['node_feature_mean'].numel(),edge_dim=n['edge_mean'].numel(),hidden_size=int(p['hidden_size']),latent_dim=int(p['latent_dim']),latent_tokens=int(p['latent_tokens']),reconstruction_dim=n['target_mean'].numel(),message_passing_steps=int(p.get('message_passing_steps',0)))
 a.set_edge_normalization(n['edge_mean'],n['edge_std'],n.get('ref_edge_mean'),n.get('ref_edge_std'));a.edge_mode=str(p.get('edge_mode','stored'));a.load_state_dict(b['ae_state_dict'],strict=True);a.eval()
 for q in a.parameters():q.requires_grad_(False)
 return a,p,n,b
def statehash(a):
 h=hashlib.sha256()
 for k,v in sorted(a.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);x=ap.parse_args();torch.set_num_threads(8);out=OUT/f's{x.seed}';out.mkdir(parents=True,exist_ok=False); started=time.time()
 data_path=ROOT/'data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt'; sims=load_dataset(data_path,coordinate_normalization='position_normalization',append_lj_indicator=True,lj_max_graph_distance=3); selected=[sims[i] for i in IDS]
 if any(len(s)!=200 or s[0].edge_attr.size(1)!=5 or int(getattr(s[0],'lj_two_hop_edges_added',0))<=0 for s in selected):raise RuntimeError('required LJ 5-channel distance3 schema failed')
 recipe={'seed':x.seed,'ids':list(IDS),'frames':list(FRAMES),'optimizer':'Adam 200 steps lr=0.02','starts':'encoded; encoded + seeded Gaussian(0.1*empirical selected-TRAIN encoded-code std, elementwise floor 1e-3)','frozen_code':str(FROZEN),'hashes':{'frozen_hashes':digest(FROZEN/'hashes.json'),'frozen_data_py':digest(FROZEN/'src/lss/data.py'),'dataset':digest(data_path),'runner':digest(Path(__file__))},'job_id':os.environ.get('PBS_JOBID'),'no_heldout_selection':True,'no_ae_weight_updates':True};(out/'recipe.json').write_text(json.dumps(recipe,indent=2))
 rows=[]; traces=[]
 for variant in VARIANTS:
  ck=HISTORICAL_RESULTS/f'compact_lj_reconstruction/{variant}_s{x.seed}_code_v2/ae.pt'; a,p,n,b=model(ck); before=statehash(a)
  with torch.no_grad(): codes=torch.stack([encode_frame_latent(a,sim,t,pos_dim=2,node_feature_mode=p['node_feature_mode'],normalizers=n,device='cpu') for sim in selected for t in range(200)])
  sigma=codes.std(0,unbiased=False).clamp_min(1e-3); g=torch.Generator().manual_seed(100000+x.seed)
  for sim,idx in zip(selected,IDS):
   for frame in FRAMES:
    target=sim[frame].x[:,:2].float()
    encoded=encode_frame_latent(a,sim,frame,pos_dim=2,node_feature_mode=p['node_feature_mode'],normalizers=n,device='cpu').detach()
    for start_name,start in [('encoded',encoded),('encoded_plus_noise',encoded+0.1*sigma*torch.randn(encoded.shape,generator=g))]:
     z=start.detach().clone().requires_grad_(True); opt=torch.optim.Adam([z],lr=.02); initial=None
     for step in range(201):
      pred=decode_latent_positions(a,sim,z,frame,pos_dim=2,ae_target_mode=p['ae_target_mode'],normalizers=n,device='cpu'); loss=((pred-target)**2).mean()
      if initial is None:initial=float(loss.detach())
      traces.append({'variant':variant,'original_index':idx,'frame':frame,'start':start_name,'step':step,'coordinate_mse':float(loss.detach())})
      if step<200:opt.zero_grad();loss.backward();opt.step()
     best=min(r['coordinate_mse'] for r in traces[-201:]);rows.append({'variant':variant,'seed':x.seed,'original_index':idx,'frame':frame,'start':start_name,'initial_encoded_mse':initial,'best_coordinate_mse':best,'final_coordinate_mse':traces[-1]['coordinate_mse'],'improvement_fraction':1-best/initial if initial>0 else None})
  after=statehash(a)
  if before!=after:raise RuntimeError('frozen AE weights changed')
  recipe.setdefault('checkpoints',{})[variant]={'path':str(ck),'sha256':digest(ck),'weights_hash_before':before,'weights_hash_after':after,'latent_std':sigma.tolist()}
 (out/'rows.json').write_text(json.dumps(rows,indent=2));(out/'loss_trace.json').write_text(json.dumps(traces));recipe.update(status='completed',seconds=time.time()-started,caveat='Fitted codes optimize against the same observed coordinates. Improvements diagnose decoder expressiveness versus encoded inference locally; they do not prove global optima, generalization, or a causal explanation.')
 (out/'recipe.json').write_text(json.dumps(recipe,indent=2));print(json.dumps({'rows':rows,'seconds':recipe['seconds']},indent=2))
if __name__=='__main__':main()
