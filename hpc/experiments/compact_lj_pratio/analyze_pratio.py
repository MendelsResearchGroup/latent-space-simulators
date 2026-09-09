"""Source-wise paired comparisons of post-fit response scores; no selection."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'notebooks/results/compact_lj_pratio/collection'
seeds=pd.read_csv(BASE/'source_frame_seed_results.csv')
summary=pd.read_csv(BASE/'source_frame_aggregate.csv')
summary[summary.frame.isin([100,199])].to_csv(BASE/'endpoint_summary.csv',index=False)
new=seeds[seeds.study=='compact_lj_spatial_decoder'].copy()
new['baseline_variant']=new.variant.str.replace('spatial_','',regex=False)
old=seeds[seeds.study=='compact_lj_reconstruction']
keys=['seed','source','split','frame']
pairs=new.merge(old,left_on=['baseline_variant',*keys],right_on=['variant',*keys],suffixes=('_spatial','_baseline'),validate='one_to_one')
pairs['r2_difference']=pairs.p_ratio_r2_spatial-pairs.p_ratio_r2_baseline
pairs['coordinate_mse_change_percent']=100*(pairs.coordinate_mse_mean_spatial/pairs.coordinate_mse_mean_baseline-1)
pairs['r2_improved']=pairs.r2_difference>0
pairs.to_csv(BASE/'spatial_paired_seed_results.csv',index=False)
paired=pairs.groupby(['variant_spatial','baseline_variant','source','split','frame']).agg(
    seeds=('seed','nunique'),baseline_r2=('p_ratio_r2_baseline','mean'),spatial_r2=('p_ratio_r2_spatial','mean'),
    r2_difference_mean=('r2_difference','mean'),r2_difference_std=('r2_difference','std'),
    improved_seeds=('r2_improved','sum'),valid_baseline=('valid_baseline','sum'),total_baseline=('total_baseline','sum'),
    valid_spatial=('valid_spatial','sum'),total_spatial=('total_spatial','sum'),
    coordinate_mse_change_percent_mean=('coordinate_mse_change_percent','mean')).reset_index()
paired.to_csv(BASE/'spatial_paired_summary.csv',index=False)
print(summary[(summary.source=='lj_noisy')&summary.frame.isin([100,199])][['variant','frame','seeds','p_ratio_r2_mean','p_ratio_r2_std','valid_sum','total_sum']].to_string(index=False))
