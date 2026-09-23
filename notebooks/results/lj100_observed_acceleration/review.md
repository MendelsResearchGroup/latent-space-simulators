# Shared propagator through step100

Validation development; final test untouched. All arms retained.
p-ratio R² is physical endpoint response, original side groups at quantile0.10.
Counts sum repeated evaluations across model seeds;20 unique networks/source.

| Family | Observed frames | Source | Mode | p-ratio R² (single seed) | Seeds | Valid/total |
| --- | --- | --- | --- | --- | --- | --- |
| observed_acceleration | 3 | depablo_low_temp | linear_history | -0.410 | 1/1 | 20/20 |
| observed_acceleration | 3 | depablo_low_temp | prefix_prediction | 0.976 | 1/1 | 20/20 |
| observed_acceleration | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| observed_acceleration | 3 | lj_noisy | linear_history | -0.900 | 1/1 | 20/20 |
| observed_acceleration | 3 | lj_noisy | prefix_prediction | -0.048 | 1/1 | 20/20 |
| observed_acceleration | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| observed_acceleration | 3 | reid | linear_history | -2.098 | 1/1 | 20/20 |
| observed_acceleration | 3 | reid | prefix_prediction | 0.926 | 1/1 | 20/20 |
| observed_acceleration | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |

## Run outcomes

 seed                family  observed_frames                      run_name    status              job_id
  123 observed_acceleration                3 observed_acceleration_h3_s123 completed 4680417.zeus-master
