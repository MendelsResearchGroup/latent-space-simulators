# Shared propagator through step100

Validation development; final test untouched. All arms retained.
p-ratio R² is physical endpoint response, original side groups at quantile0.10.
Counts sum repeated evaluations across model seeds;20 unique networks/source.

| Family | Context | Source | Mode | p-ratio R² mean ± SD | Seeds | Valid/total |
| --- | --- | --- | --- | --- | --- | --- |
| graph_operator | attention | depablo_low_temp | direct_prediction | 0.954 ± 0.008 | 3/3 | 60/60 |
| graph_operator | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| graph_operator | attention | depablo_mixed_temp | direct_prediction | 0.915 ± 0.019 | 3/3 | 60/60 |
| graph_operator | attention | depablo_mixed_temp | reconstruction | 0.871 ± 0.043 | 3/3 | 60/60 |
| graph_operator | attention | lj_noisy | direct_prediction | -0.301 ± 0.147 | 3/3 | 60/60 |
| graph_operator | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| graph_operator | attention | reid | direct_prediction | 0.854 ± 0.071 | 3/3 | 60/60 |
| graph_operator | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| soft_experts | attention | depablo_low_temp | direct_prediction | 0.941 ± 0.009 | 3/3 | 60/60 |
| soft_experts | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| soft_experts | attention | depablo_mixed_temp | direct_prediction | 0.902 ± 0.029 | 3/3 | 60/60 |
| soft_experts | attention | depablo_mixed_temp | reconstruction | 0.871 ± 0.043 | 3/3 | 60/60 |
| soft_experts | attention | lj_noisy | direct_prediction | -0.134 ± 0.012 | 3/3 | 60/60 |
| soft_experts | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| soft_experts | attention | reid | direct_prediction | 0.869 ± 0.012 | 3/3 | 60/60 |
| soft_experts | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| trajectory_flow | attention | depablo_low_temp | ensemble_mean | 0.106 ± 0.129 | 3/3 | 60/60 |
| trajectory_flow | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| trajectory_flow | attention | depablo_mixed_temp | ensemble_mean | 0.052 ± 0.182 | 3/3 | 60/60 |
| trajectory_flow | attention | depablo_mixed_temp | reconstruction | 0.871 ± 0.043 | 3/3 | 60/60 |
| trajectory_flow | attention | lj_noisy | ensemble_mean | -0.318 ± 0.050 | 3/3 | 60/60 |
| trajectory_flow | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| trajectory_flow | attention | reid | ensemble_mean | 0.150 ± 0.120 | 3/3 | 60/60 |
| trajectory_flow | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |

## Mixed-T prefix linear-fit response at100

         family              kind  p_ratio_linear_fit_r2_mean  p_ratio_linear_fit_r2_std  linear_fit_valid  p_ratio_total
 graph_operator direct_prediction                    0.928521                   0.024621                60             60
 graph_operator    reconstruction                   -2.721001                   6.015959                60             60
   soft_experts direct_prediction                    0.892882                   0.019584                60             60
   soft_experts    reconstruction                   -2.721001                   6.015959                60             60
trajectory_flow     ensemble_mean                    0.060265                   0.132820                60             60
trajectory_flow    reconstruction                   -2.721001                   6.015959                60             60

## Run outcomes

 seed          family context_mode                       run_name    status              job_id transfer_status
  123  graph_operator    attention  graph_operator_attention_s123 completed 4678881.zeus-master       completed
  456  graph_operator    attention  graph_operator_attention_s456 completed 4678882.zeus-master       completed
  786  graph_operator    attention  graph_operator_attention_s786 completed 4678883.zeus-master       completed
  123    soft_experts    attention    soft_experts_attention_s123 completed 4678884.zeus-master       completed
  456    soft_experts    attention    soft_experts_attention_s456 completed 4678885.zeus-master       completed
  786    soft_experts    attention    soft_experts_attention_s786 completed 4678886.zeus-master       completed
  123 trajectory_flow    attention trajectory_flow_attention_s123 completed 4678887.zeus-master       completed
  456 trajectory_flow    attention trajectory_flow_attention_s456 completed 4678888.zeus-master       completed
  786 trajectory_flow    attention trajectory_flow_attention_s786 completed 4678889.zeus-master       completed
