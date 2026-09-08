"""Post-fit coordinate comparisons; no automatic model promotion or expert targets."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(os.environ.get('LSS_PROJECT_ROOT',Path(__file__).resolve().parents[3]))
HISTORICAL_RESULTS=Path(os.environ.get('LSS_HISTORICAL_RESULTS_ROOT',ROOT/'notebooks/results'))
RESULTS_ROOT=Path(os.environ.get('LSS_RESULTS_ROOT',ROOT/'notebooks/results'))
BASE=HISTORICAL_RESULTS/'compact_lj_reconstruction'
VERSION=os.environ.get('LSS_CODE_VERSION','code_v2')
OUT=RESULTS_ROOT/f'hpc_diagnostics/compact_lj_reconstruction/comparisons_{VERSION}'
OUT.mkdir(parents=True,exist_ok=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
ledger=BASE/'jobs.jsonl'
accepted={}
for line in ledger.read_text().splitlines():
    j=json.loads(line)
    if j.get('kind')=='run' and j.get('job_id') and not j.get('smoke') and j.get('code_version')==VERSION:
        accepted[(j['variant'],j['seed'])]=j
states=[];frames=[];inputs={str(ledger):sha(ledger)}
for (variant,seed),j in accepted.items():
    d=BASE/f'{variant}_s{seed}_{VERSION}'
    status='completed' if (d/'completed.json').exists() else 'failed' if (d/'failed.json').exists() else 'incomplete'
    states.append(dict(variant=variant,seed=seed,job_id=j['job_id'],status=status))
    if status!='completed': continue
    p=d/'source_frame_rows.csv';inputs[str(p)]=sha(p)
    f=pd.read_csv(p);f['variant']=variant;f['seed']=seed;frames.append(f)
pd.DataFrame(states).to_csv(OUT/'run_status.csv',index=False)
if frames:
    data=pd.concat(frames,ignore_index=True)
    audit_path=HISTORICAL_RESULTS/'compact_lj_motion_audit/frame_rows_v1.json'
    inputs[str(audit_path)]=sha(audit_path)
    audit=pd.read_json(audit_path)
    train=data[data.split=='train'].merge(audit,on=['source','original_index','frame'],validate='many_to_one',how='left',indicator=True)
    train['matched_motion_audit']=train['_merge']=='both'
    train.to_csv(OUT/'matched_train_motion_rows.csv',index=False)
    metrics=['coordinate_mse','zero_motion_mse','affine_x0_to_xt_mse','pca_in_sample_rank4_mse','shuffled_same_source_coordinate_mse']
    valid=train[(train.frame>0)&train.matched_motion_audit]
    summary=valid.groupby(['variant','source','frame']).agg(seeds=('seed','nunique'),unique_networks=('original_index','nunique'),evaluations=('valid','size'),coordinate_valid=('valid','sum'),**{k:(k,'mean') for k in metrics}).reset_index()
    summary['ae_over_affine_mse']=summary.coordinate_mse/summary.affine_x0_to_xt_mse.replace(0,np.nan)
    summary['shuffled_over_ae_mse']=summary.shuffled_same_source_coordinate_mse/summary.coordinate_mse.replace(0,np.nan)
    summary.to_csv(OUT/'matched_train_motion_summary.csv',index=False)
    paired=[]
    for dim in [4,8]:
        for split in ['train','validation']:
            f=data[(data.source=='lj_noisy')&(data.split==split)]
            a=f[f.variant==f'r16_d{dim}'];b=f[f.variant==f'ljonly_r16_d{dim}']
            pair=a.merge(b,on=['source','split','seed','original_index','frame'],suffixes=('_shared','_ljonly'),validate='one_to_one')
            pair['latent_dim']=dim
            pair['shared_minus_ljonly_mse']=pair.coordinate_mse_shared-pair.coordinate_mse_ljonly
            paired.append(pair)
    paired=pd.concat(paired,ignore_index=True)
    paired.to_csv(OUT/'shared_ljonly_paired_rows.csv',index=False)
    if not paired.empty:
        seed=paired.groupby(['latent_dim','split','source','frame','seed']).agg(networks=('original_index','nunique'),shared_mse=('coordinate_mse_shared','mean'),ljonly_mse=('coordinate_mse_ljonly','mean'),difference=('shared_minus_ljonly_mse','mean')).reset_index()
        seed.to_csv(OUT/'shared_ljonly_paired_seed_results.csv',index=False)
        seed.groupby(['latent_dim','split','source','frame']).agg(seeds=('seed','nunique'),shared_mse=('shared_mse','mean'),ljonly_mse=('ljonly_mse','mean'),difference_mean=('difference','mean'),difference_std=('difference','std')).reset_index().to_csv(OUT/'shared_ljonly_summary.csv',index=False)
(OUT/'provenance.json').write_text(json.dumps(dict(job_id=os.environ.get('PBS_JOBID'),code_version=VERSION,script_sha256=sha(Path(__file__)),inputs=inputs,declared_runs=50,accepted_runs=len(accepted),completed_runs=sum(s['status']=='completed' for s in states),caveats=['Affine and PCA are optimistic fits to observed training targets, not forecasts or training losses.','LJ-only and shared fitting differ in normalization and optimizer exposure; paired outcomes do not isolate causal interference.','Training rows are diagnostics, validation rows are not final test; no mixed-T or expert metric selects models.']),indent=2))
print(OUT,flush=True)
