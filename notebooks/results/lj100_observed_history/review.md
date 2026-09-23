# Shared propagator through step100

Validation development; final test untouched. All arms retained.
p-ratio R² is physical endpoint response, original side groups at quantile0.10.
Counts sum repeated evaluations across model seeds;20 unique networks/source.

| Family | Observed frames | Source | Mode | p-ratio R² mean ± SD | Seeds | Valid/total |
| --- | --- | --- | --- | --- | --- | --- |
| history_mlp | 1 | depablo_low_temp | prefix_prediction | 0.943 ± 0.010 | 3/3 | 60/60 |
| history_mlp | 1 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| history_mlp | 1 | lj_noisy | prefix_prediction | -0.082 ± 0.104 | 3/3 | 60/60 |
| history_mlp | 1 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| history_mlp | 1 | reid | prefix_prediction | 0.879 ± 0.020 | 3/3 | 60/60 |
| history_mlp | 1 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| history_mlp | 3 | depablo_low_temp | linear_history | -0.980 ± 0.609 | 3/3 | 60/60 |
| history_mlp | 3 | depablo_low_temp | prefix_prediction | 0.963 ± 0.027 | 3/3 | 60/60 |
| history_mlp | 3 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| history_mlp | 3 | lj_noisy | linear_history | -26.824 ± 24.098 | 3/3 | 60/60 |
| history_mlp | 3 | lj_noisy | prefix_prediction | -0.000 ± 0.260 | 3/3 | 60/60 |
| history_mlp | 3 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| history_mlp | 3 | reid | linear_history | -4.998 ± 3.321 | 3/3 | 60/60 |
| history_mlp | 3 | reid | prefix_prediction | 0.914 ± 0.012 | 3/3 | 60/60 |
| history_mlp | 3 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| velocity_residual | 1 | depablo_low_temp | prefix_prediction | 0.943 ± 0.010 | 3/3 | 60/60 |
| velocity_residual | 1 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| velocity_residual | 1 | lj_noisy | prefix_prediction | -0.082 ± 0.104 | 3/3 | 60/60 |
| velocity_residual | 1 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| velocity_residual | 1 | reid | prefix_prediction | 0.879 ± 0.020 | 3/3 | 60/60 |
| velocity_residual | 1 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| velocity_residual | 3 | depablo_low_temp | linear_history | -0.980 ± 0.609 | 3/3 | 60/60 |
| velocity_residual | 3 | depablo_low_temp | prefix_prediction | 0.954 ± 0.022 | 3/3 | 60/60 |
| velocity_residual | 3 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| velocity_residual | 3 | lj_noisy | linear_history | -26.824 ± 24.098 | 3/3 | 60/60 |
| velocity_residual | 3 | lj_noisy | prefix_prediction | -0.204 ± 0.148 | 3/3 | 60/60 |
| velocity_residual | 3 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| velocity_residual | 3 | reid | linear_history | -4.998 ± 3.321 | 3/3 | 60/60 |
| velocity_residual | 3 | reid | prefix_prediction | 0.885 ± 0.004 | 3/3 | 60/60 |
| velocity_residual | 3 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |

## Run outcomes

 seed            family  observed_frames                  run_name    status              job_id
  123       history_mlp                1       history_mlp_h1_s123 completed 4679056.zeus-master
  456       history_mlp                1       history_mlp_h1_s456 completed 4679057.zeus-master
  786       history_mlp                1       history_mlp_h1_s786 completed 4679058.zeus-master
  123       history_mlp                3       history_mlp_h3_s123 completed 4679059.zeus-master
  456       history_mlp                3       history_mlp_h3_s456 completed 4679060.zeus-master
  786       history_mlp                3       history_mlp_h3_s786 completed 4679061.zeus-master
  123 velocity_residual                1 velocity_residual_h1_s123 completed 4679062.zeus-master
  456 velocity_residual                1 velocity_residual_h1_s456 completed 4679063.zeus-master
  786 velocity_residual                1 velocity_residual_h1_s786 completed 4679064.zeus-master
  123 velocity_residual                3 velocity_residual_h3_s123 completed 4679065.zeus-master
  456 velocity_residual                3 velocity_residual_h3_s456 completed 4679066.zeus-master
  786 velocity_residual                3 velocity_residual_h3_s786 completed 4679067.zeus-master
