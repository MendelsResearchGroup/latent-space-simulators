# Shared propagator through step100

Validation development; final test untouched. All arms retained.
p-ratio R² is physical endpoint response, original side groups at quantile0.10.
Counts sum repeated evaluations across model seeds;20 unique networks/source.

| Objective | LR | Source | Mode | p-ratio R² mean ± SD | Seeds | Valid/total |
| --- | --- | --- | --- | --- | --- | --- |
| decoded_positions | 0.0001 | depablo_low_temp | autonomous | 0.933 ± 0.043 | 3/3 | 60/60 |
| decoded_positions | 0.0001 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| decoded_positions | 0.0001 | lj_noisy | autonomous | -0.624 ± 0.240 | 3/3 | 60/60 |
| decoded_positions | 0.0001 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| decoded_positions | 0.0001 | reid | autonomous | 0.817 ± 0.026 | 3/3 | 60/60 |
| decoded_positions | 0.0001 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| decoded_positions | 0.0003 | depablo_low_temp | autonomous | 0.934 ± 0.032 | 3/3 | 60/60 |
| decoded_positions | 0.0003 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| decoded_positions | 0.0003 | lj_noisy | autonomous | -0.330 ± 0.133 | 3/3 | 60/60 |
| decoded_positions | 0.0003 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| decoded_positions | 0.0003 | reid | autonomous | 0.835 ± 0.018 | 3/3 | 60/60 |
| decoded_positions | 0.0003 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| latent_state | 0.0001 | depablo_low_temp | autonomous | 0.934 ± 0.039 | 3/3 | 60/60 |
| latent_state | 0.0001 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| latent_state | 0.0001 | lj_noisy | autonomous | -0.495 ± 0.274 | 3/3 | 60/60 |
| latent_state | 0.0001 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| latent_state | 0.0001 | reid | autonomous | 0.814 ± 0.025 | 3/3 | 60/60 |
| latent_state | 0.0001 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |
| latent_state | 0.0003 | depablo_low_temp | autonomous | 0.939 ± 0.029 | 3/3 | 60/60 |
| latent_state | 0.0003 | depablo_low_temp | reconstruction | 0.992 ± 0.000 | 3/3 | 60/60 |
| latent_state | 0.0003 | lj_noisy | autonomous | -0.573 ± 0.529 | 3/3 | 60/60 |
| latent_state | 0.0003 | lj_noisy | reconstruction | 0.594 ± 0.064 | 3/3 | 60/60 |
| latent_state | 0.0003 | reid | autonomous | 0.820 ± 0.026 | 3/3 | 60/60 |
| latent_state | 0.0003 | reid | reconstruction | 0.957 ± 0.010 | 3/3 | 60/60 |

## Run outcomes

 seed         objective     lr                             run_name    status              job_id
  123      latent_state 0.0001      mp2d_s123_latent_state_lr0.0001 completed 4678481.zeus-master
  123      latent_state 0.0003      mp2d_s123_latent_state_lr0.0003 completed 4678482.zeus-master
  123 decoded_positions 0.0001 mp2d_s123_decoded_positions_lr0.0001 completed 4678483.zeus-master
  123 decoded_positions 0.0003 mp2d_s123_decoded_positions_lr0.0003 completed 4678484.zeus-master
  456      latent_state 0.0001      mp2d_s456_latent_state_lr0.0001 completed 4678485.zeus-master
  456      latent_state 0.0003      mp2d_s456_latent_state_lr0.0003 completed 4678486.zeus-master
  456 decoded_positions 0.0001 mp2d_s456_decoded_positions_lr0.0001 completed 4678487.zeus-master
  456 decoded_positions 0.0003 mp2d_s456_decoded_positions_lr0.0003 completed 4678488.zeus-master
  786      latent_state 0.0001      mp2d_s786_latent_state_lr0.0001 completed 4678489.zeus-master
  786      latent_state 0.0003      mp2d_s786_latent_state_lr0.0003 completed 4678490.zeus-master
  786 decoded_positions 0.0001 mp2d_s786_decoded_positions_lr0.0001 completed 4678491.zeus-master
  786 decoded_positions 0.0003 mp2d_s786_decoded_positions_lr0.0003 completed 4678492.zeus-master
