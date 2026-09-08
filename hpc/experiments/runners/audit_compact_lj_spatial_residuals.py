"""Affine/non-affine coordinate residual audit for frozen compact-LJ AEs."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(os.environ.get('LSS_PROJECT_ROOT',Path(__file__).resolve().parents[3])); RESULTS_ROOT=Path(os.environ.get('LSS_RESULTS_ROOT',ROOT/'notebooks/results')); HISTORICAL_RESULTS=Path(os.environ.get('LSS_HISTORICAL_RESULTS_ROOT',ROOT/'notebooks/results')); FROZEN=HISTORICAL_RESULTS/'compact_lj_reconstruction/code_v2';sys.path.insert(0,str(FROZEN/'src'))
import numpy as np, torch
from lss.data import load_dataset
from lss.latent.experiment import _autoencoder_class
from lss.latent.training import encode_frame_latent,decode_latent_positions
OUT=RESULTS_ROOT/'hpc_diagnostics/compact_lj_spatial_residuals'; MANIFEST=Path(os.environ.get('LSS_SPLIT_MANIFEST',HISTORICAL_RESULTS/'lj_ae_08_bridge/split_manifest.json')); VARIANTS=('r16_d4','mp2_r16_d8','ljonly_r16_d4');SEEDS=(123,456,786);FRAMES=(25,100,199)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def restore(path):
 b=torch.load(path,map_location='cpu',weights_only=False);p={**b['params'],**b['source_spec']};s=b['stats'];n={k:s[k].float() for k in ('target_mean','target_std','node_feature_mean','node_feature_std','edge_mean','edge_std')}
 for k in ('ref_edge_mean','ref_edge_std'):
  if k in s:n[k]=s[k].float()
 a=_autoencoder_class(str(p.get('autoencoder_model','attention')).lower())(pos_dim=int(p['pos_dim']),node_feature_dim=n['node_feature_mean'].numel(),edge_dim=n['edge_mean'].numel(),hidden_size=int(p['hidden_size']),latent_dim=int(p['latent_dim']),latent_tokens=int(p['latent_tokens']),reconstruction_dim=n['target_mean'].numel(),message_passing_steps=int(p.get('message_passing_steps',0)))
 a.set_edge_normalization(n['edge_mean'],n['edge_std'],n.get('ref_edge_mean'),n.get('ref_edge_std'));a.edge_mode=str(p.get('edge_mode','stored'));a.load_state_dict(b['ae_state_dict'],strict=True);a.eval()
 for q in a.parameters():q.requires_grad_(False)
 return a,p,n
def main():
 OUT.mkdir(parents=True,exist_ok=True);m=json.loads(MANIFEST.read_text());[spec.update(path=str(ROOT/'data'/Path(spec['path']).name)) for spec in m.get('sources',{}).values()];ids={'train':m['sources']['lj_noisy']['split_indices']['train'][:5],'validation':m['sources']['lj_noisy']['split_indices']['val'][:20]};data_path=Path(m['sources']['lj_noisy']['path']);sims=load_dataset(data_path,coordinate_normalization='position_normalization',append_lj_indicator=True,lj_max_graph_distance=3)
 if any(sims[i][0].edge_attr.size(1)!=5 or int(getattr(sims[i][0],'lj_two_hop_edges_added',0))<=0 for v in ids.values() for i in v):raise RuntimeError('LJ distance3 five-channel preflight failed')
 rows=[];checks={};started=time.time()
 for variant in VARIANTS:
  for seed in SEEDS:
   ck=HISTORICAL_RESULTS/f'compact_lj_reconstruction/{variant}_s{seed}_code_v2/ae.pt';a,p,n=restore(ck);checks[f'{variant}_s{seed}']={'path':str(ck),'sha256':sha(ck)}
   with torch.no_grad():
    for split,indices in ids.items():
     for index in indices:
      sim=sims[index];x0=sim[0].x[:,:2].float();design=torch.cat([x0,torch.ones((x0.size(0),1))],1)
      for frame in FRAMES:
       z=encode_frame_latent(a,sim,frame,pos_dim=2,node_feature_mode=p['node_feature_mode'],normalizers=n,device='cpu');pred=decode_latent_positions(a,sim,z,frame,pos_dim=2,ae_target_mode=p['ae_target_mode'],normalizers=n,device='cpu');true=sim[frame].x[:,:2].float();td=true-x0;pd=pred-x0
       # Columns span all node-wise affine displacement fields; torch lstsq yields the orthogonal projection.
       ta=torch.linalg.lstsq(design,td).solution;pa=torch.linalg.lstsq(design,pd).solution;taf=design@ta;paf=design@pa;tr=td-taf;pr=pd-paf
       total=((pred-true)**2).mean();ae=((paf-taf)**2).mean();ne=((pr-tr)**2).mean();de=float((total-ae-ne).abs());den=float(torch.linalg.vector_norm(tr)*torch.linalg.vector_norm(pr));cos=float((tr*pr).sum()/den) if den>1e-15 else float('nan')
       rows.append({'variant':variant,'seed':seed,'split':split,'original_index':index,'frame':frame,'nodes':int(x0.size(0)),'coordinate_mse':float(total),'true_nonaffine_energy_mse':float((tr**2).mean()),'predicted_nonaffine_energy_mse':float((pr**2).mean()),'nonaffine_residual_error_mse':float(ne),'affine_component_error_mse':float(ae),'residual_cosine':cos,'predicted_to_true_nonaffine_energy_ratio':float((pr**2).mean()/((tr**2).mean())) if float((tr**2).mean())>0 else float('nan'),'decomposition_absolute_error':de,'decomposition_pass':bool(de<1e-8)})
 rec={'status':'completed','job_id':os.environ.get('PBS_JOBID'),'seconds':time.time()-started,'ids':ids,'frames':FRAMES,'checkpoints':checks,'hashes':{'manifest':sha(MANIFEST),'dataset':sha(data_path),'frozen_hashes':sha(FROZEN/'hashes.json'),'frozen_data_py':sha(FROZEN/'src/lss/data.py'),'runner':sha(Path(__file__))},'caveat':'Coordinate-only post-fit diagnostic. Affine projections and residual quantities are not training targets, model selection criteria, or evidence of causal mechanism/generalization.'}
 (OUT/'rows.json').write_text(json.dumps(rows,indent=2));
 summary={}
 for split in ids:
  for frame in FRAMES:
   q=[r for r in rows if r['split']==split and r['frame']==frame]
   summary[f'{split}_f{frame}']={k:float(np.nanmean([r[k] for r in q])) for k in ('coordinate_mse','true_nonaffine_energy_mse','predicted_nonaffine_energy_mse','nonaffine_residual_error_mse','affine_component_error_mse','residual_cosine','predicted_to_true_nonaffine_energy_ratio')};summary[f'{split}_f{frame}'].update(valid=len(q),total=len(ids[split])*len(VARIANTS)*len(SEEDS),decomposition_pass=sum(r['decomposition_pass'] for r in q))
 rec['pooled_summary']=summary;(OUT/'recipe.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
