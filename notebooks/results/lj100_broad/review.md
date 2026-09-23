# Shared propagator through step100

Validation development; final test untouched. All arms retained.
p-ratio R² is physical endpoint response, original side groups at quantile0.10.
Counts sum repeated evaluations across model seeds;20 unique networks/source.

| Family | Context | Source | Mode | p-ratio R² mean ± SD | Seeds | Valid/total |
| --- | --- | --- | --- | --- | --- | --- |
| cubic | attention | depablo_low_temp | direct_prediction | 0.132 ± 0.172 | 3/3 | 60/60 |
| cubic | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| cubic | attention | lj_noisy | direct_prediction | -0.198 ± 0.047 | 3/3 | 60/60 |
| cubic | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| cubic | attention | reid | direct_prediction | 0.443 ± 0.313 | 3/3 | 60/60 |
| cubic | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| cubic | mean | depablo_low_temp | direct_prediction | 0.140 ± 0.169 | 3/3 | 60/60 |
| cubic | mean | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| cubic | mean | lj_noisy | direct_prediction | -0.111 ± 0.133 | 3/3 | 60/60 |
| cubic | mean | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| cubic | mean | reid | direct_prediction | 0.207 ± 0.109 | 3/3 | 60/60 |
| cubic | mean | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| direct_mlp | attention | depablo_low_temp | direct_prediction | 0.163 ± 0.165 | 3/3 | 60/60 |
| direct_mlp | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| direct_mlp | attention | lj_noisy | direct_prediction | -0.392 ± 0.237 | 3/3 | 60/60 |
| direct_mlp | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| direct_mlp | attention | reid | direct_prediction | 0.285 ± 0.299 | 3/3 | 60/60 |
| direct_mlp | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| direct_mlp | mean | depablo_low_temp | direct_prediction | 0.179 ± 0.156 | 3/3 | 60/60 |
| direct_mlp | mean | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| direct_mlp | mean | lj_noisy | direct_prediction | -0.079 ± 0.172 | 3/3 | 60/60 |
| direct_mlp | mean | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| direct_mlp | mean | reid | direct_prediction | 0.113 ± 0.116 | 3/3 | 60/60 |
| direct_mlp | mean | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| gru | attention | depablo_low_temp | autonomous | 0.481 ± 0.499 | 3/3 | 60/60 |
| gru | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| gru | attention | lj_noisy | autonomous | -0.165 ± 0.052 | 3/3 | 60/60 |
| gru | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| gru | attention | reid | autonomous | 0.398 ± 0.427 | 3/3 | 60/60 |
| gru | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| gru | mean | depablo_low_temp | autonomous | 0.289 ± 0.268 | 3/3 | 60/60 |
| gru | mean | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| gru | mean | lj_noisy | autonomous | -0.185 ± 0.083 | 3/3 | 60/60 |
| gru | mean | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| gru | mean | reid | autonomous | 0.166 ± 0.294 | 3/3 | 60/60 |
| gru | mean | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| history_attention | attention | depablo_low_temp | autonomous | 0.072 ± 0.106 | 3/3 | 60/60 |
| history_attention | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| history_attention | attention | lj_noisy | autonomous | -0.140 ± 0.078 | 3/3 | 60/60 |
| history_attention | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| history_attention | attention | reid | autonomous | 0.870 ± 0.053 | 3/3 | 60/60 |
| history_attention | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| history_attention | mean | depablo_low_temp | autonomous | 0.143 ± 0.237 | 3/3 | 60/60 |
| history_attention | mean | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| history_attention | mean | lj_noisy | autonomous | 0.014 ± 0.176 | 3/3 | 60/60 |
| history_attention | mean | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| history_attention | mean | reid | autonomous | 0.542 ± 0.575 | 3/3 | 60/60 |
| history_attention | mean | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| inertial | attention | depablo_low_temp | autonomous | 0.122 ± 0.130 | 3/3 | 60/60 |
| inertial | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| inertial | attention | lj_noisy | autonomous | -0.212 ± 0.112 | 3/3 | 60/60 |
| inertial | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| inertial | attention | reid | autonomous | 0.466 ± 0.338 | 3/3 | 60/60 |
| inertial | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| inertial | mean | depablo_low_temp | autonomous | 0.154 ± 0.173 | 3/3 | 60/60 |
| inertial | mean | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| inertial | mean | lj_noisy | autonomous | -0.091 ± 0.148 | 3/3 | 60/60 |
| inertial | mean | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| inertial | mean | reid | autonomous | 0.193 ± 0.103 | 3/3 | 60/60 |
| inertial | mean | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| residual_mlp | attention | depablo_low_temp | autonomous | 0.136 ± 0.160 | 3/3 | 60/60 |
| residual_mlp | attention | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| residual_mlp | attention | lj_noisy | autonomous | -0.400 ± 0.362 | 3/3 | 60/60 |
| residual_mlp | attention | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| residual_mlp | attention | reid | autonomous | 0.436 ± 0.298 | 3/3 | 60/60 |
| residual_mlp | attention | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| residual_mlp | mean | depablo_low_temp | autonomous | 0.121 ± 0.146 | 3/3 | 60/60 |
| residual_mlp | mean | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| residual_mlp | mean | lj_noisy | autonomous | -0.073 ± 0.167 | 3/3 | 60/60 |
| residual_mlp | mean | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| residual_mlp | mean | reid | autonomous | 0.185 ± 0.098 | 3/3 | 60/60 |
| residual_mlp | mean | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |

## Run outcomes

 seed            family context_mode                         run_name    status              job_id
  123      residual_mlp         mean           residual_mlp_mean_s123 completed 4678657.zeus-master
  456      residual_mlp         mean           residual_mlp_mean_s456 completed 4678658.zeus-master
  786      residual_mlp         mean           residual_mlp_mean_s786 completed 4678659.zeus-master
  123      residual_mlp    attention      residual_mlp_attention_s123 completed 4678660.zeus-master
  456      residual_mlp    attention      residual_mlp_attention_s456 completed 4678661.zeus-master
  786      residual_mlp    attention      residual_mlp_attention_s786 completed 4678662.zeus-master
  123          inertial         mean               inertial_mean_s123 completed 4678663.zeus-master
  456          inertial         mean               inertial_mean_s456 completed 4678664.zeus-master
  786          inertial         mean               inertial_mean_s786 completed 4678665.zeus-master
  123          inertial    attention          inertial_attention_s123 completed 4678666.zeus-master
  456          inertial    attention          inertial_attention_s456 completed 4678667.zeus-master
  786          inertial    attention          inertial_attention_s786 completed 4678668.zeus-master
  123               gru         mean                    gru_mean_s123 completed 4678669.zeus-master
  456               gru         mean                    gru_mean_s456 completed 4678670.zeus-master
  786               gru         mean                    gru_mean_s786 completed 4678671.zeus-master
  123               gru    attention               gru_attention_s123 completed 4678672.zeus-master
  456               gru    attention               gru_attention_s456 completed 4678673.zeus-master
  786               gru    attention               gru_attention_s786 completed 4678674.zeus-master
  123 history_attention         mean      history_attention_mean_s123 completed 4678675.zeus-master
  456 history_attention         mean      history_attention_mean_s456 completed 4678676.zeus-master
  786 history_attention         mean      history_attention_mean_s786 completed 4678677.zeus-master
  123 history_attention    attention history_attention_attention_s123 completed 4678678.zeus-master
  456 history_attention    attention history_attention_attention_s456 completed 4678679.zeus-master
  786 history_attention    attention history_attention_attention_s786 completed 4678680.zeus-master
  123        direct_mlp         mean             direct_mlp_mean_s123 completed 4678681.zeus-master
  456        direct_mlp         mean             direct_mlp_mean_s456 completed 4678682.zeus-master
  786        direct_mlp         mean             direct_mlp_mean_s786 completed 4678683.zeus-master
  123        direct_mlp    attention        direct_mlp_attention_s123 completed 4678684.zeus-master
  456        direct_mlp    attention        direct_mlp_attention_s456 completed 4678685.zeus-master
  786        direct_mlp    attention        direct_mlp_attention_s786 completed 4678686.zeus-master
  123             cubic         mean                  cubic_mean_s123 completed 4678687.zeus-master
  456             cubic         mean                  cubic_mean_s456 completed 4678688.zeus-master
  786             cubic         mean                  cubic_mean_s786 completed 4678689.zeus-master
  123             cubic    attention             cubic_attention_s123 completed 4678690.zeus-master
  456             cubic    attention             cubic_attention_s456 completed 4678691.zeus-master
  786             cubic    attention             cubic_attention_s786 completed 4678692.zeus-master
