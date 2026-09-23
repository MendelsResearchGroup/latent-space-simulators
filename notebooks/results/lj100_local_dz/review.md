# Shared propagator through step100

Validation development; final test untouched. All arms retained.
p-ratio R² is physical endpoint response, original side groups at quantile0.10.
Counts sum repeated evaluations across model seeds;20 unique networks/source.

| Family | Observed frames | Source | Mode | p-ratio R² (single seed) | Seeds | Valid/total |
| --- | --- | --- | --- | --- | --- | --- |
| state128 | 3 | depablo_low_temp | persistence_one_step | 0.992 | 1/1 | 20/20 |
| state128 | 3 | depablo_low_temp | prefix_prediction | 0.839 | 1/1 | 20/20 |
| state128 | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| state128 | 3 | depablo_low_temp | teacher_one_step | 0.992 | 1/1 | 20/20 |
| state128 | 3 | depablo_low_temp | velocity_one_step | 0.992 | 1/1 | 20/20 |
| state128 | 3 | lj_noisy | persistence_one_step | 0.518 | 1/1 | 20/20 |
| state128 | 3 | lj_noisy | prefix_prediction | -5.118 | 1/1 | 20/20 |
| state128 | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| state128 | 3 | lj_noisy | teacher_one_step | 0.519 | 1/1 | 20/20 |
| state128 | 3 | lj_noisy | velocity_one_step | 0.521 | 1/1 | 20/20 |
| state128 | 3 | reid | persistence_one_step | 0.951 | 1/1 | 20/20 |
| state128 | 3 | reid | prefix_prediction | 0.396 | 1/1 | 20/20 |
| state128 | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |
| state128 | 3 | reid | teacher_one_step | 0.952 | 1/1 | 20/20 |
| state128 | 3 | reid | velocity_one_step | 0.952 | 1/1 | 20/20 |
| state32 | 3 | depablo_low_temp | persistence_one_step | 0.992 | 1/1 | 20/20 |
| state32 | 3 | depablo_low_temp | prefix_prediction | 0.668 | 1/1 | 20/20 |
| state32 | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| state32 | 3 | depablo_low_temp | teacher_one_step | 0.992 | 1/1 | 20/20 |
| state32 | 3 | depablo_low_temp | velocity_one_step | 0.992 | 1/1 | 20/20 |
| state32 | 3 | lj_noisy | persistence_one_step | 0.518 | 1/1 | 20/20 |
| state32 | 3 | lj_noisy | prefix_prediction | -1.987 | 1/1 | 20/20 |
| state32 | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| state32 | 3 | lj_noisy | teacher_one_step | 0.520 | 1/1 | 20/20 |
| state32 | 3 | lj_noisy | velocity_one_step | 0.521 | 1/1 | 20/20 |
| state32 | 3 | reid | persistence_one_step | 0.951 | 1/1 | 20/20 |
| state32 | 3 | reid | prefix_prediction | 0.214 | 1/1 | 20/20 |
| state32 | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |
| state32 | 3 | reid | teacher_one_step | 0.951 | 1/1 | 20/20 |
| state32 | 3 | reid | velocity_one_step | 0.952 | 1/1 | 20/20 |
| state64 | 3 | depablo_low_temp | persistence_one_step | 0.992 | 1/1 | 20/20 |
| state64 | 3 | depablo_low_temp | prefix_prediction | 0.853 | 1/1 | 20/20 |
| state64 | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| state64 | 3 | depablo_low_temp | teacher_one_step | 0.992 | 1/1 | 20/20 |
| state64 | 3 | depablo_low_temp | velocity_one_step | 0.992 | 1/1 | 20/20 |
| state64 | 3 | lj_noisy | persistence_one_step | 0.518 | 1/1 | 20/20 |
| state64 | 3 | lj_noisy | prefix_prediction | -2.779 | 1/1 | 20/20 |
| state64 | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| state64 | 3 | lj_noisy | teacher_one_step | 0.519 | 1/1 | 20/20 |
| state64 | 3 | lj_noisy | velocity_one_step | 0.521 | 1/1 | 20/20 |
| state64 | 3 | reid | persistence_one_step | 0.951 | 1/1 | 20/20 |
| state64 | 3 | reid | prefix_prediction | 0.350 | 1/1 | 20/20 |
| state64 | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |
| state64 | 3 | reid | teacher_one_step | 0.952 | 1/1 | 20/20 |
| state64 | 3 | reid | velocity_one_step | 0.952 | 1/1 | 20/20 |
| velocity128 | 3 | depablo_low_temp | persistence_one_step | 0.992 | 1/1 | 20/20 |
| velocity128 | 3 | depablo_low_temp | prefix_prediction | 0.565 | 1/1 | 20/20 |
| velocity128 | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| velocity128 | 3 | depablo_low_temp | teacher_one_step | 0.992 | 1/1 | 20/20 |
| velocity128 | 3 | depablo_low_temp | velocity_one_step | 0.992 | 1/1 | 20/20 |
| velocity128 | 3 | lj_noisy | persistence_one_step | 0.518 | 1/1 | 20/20 |
| velocity128 | 3 | lj_noisy | prefix_prediction | -14.032 | 1/1 | 20/20 |
| velocity128 | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| velocity128 | 3 | lj_noisy | teacher_one_step | 0.520 | 1/1 | 20/20 |
| velocity128 | 3 | lj_noisy | velocity_one_step | 0.521 | 1/1 | 20/20 |
| velocity128 | 3 | reid | persistence_one_step | 0.951 | 1/1 | 20/20 |
| velocity128 | 3 | reid | prefix_prediction | 0.236 | 1/1 | 20/20 |
| velocity128 | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |
| velocity128 | 3 | reid | teacher_one_step | 0.952 | 1/1 | 20/20 |
| velocity128 | 3 | reid | velocity_one_step | 0.952 | 1/1 | 20/20 |
| velocity64 | 3 | depablo_low_temp | persistence_one_step | 0.992 | 1/1 | 20/20 |
| velocity64 | 3 | depablo_low_temp | prefix_prediction | 0.343 | 1/1 | 20/20 |
| velocity64 | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| velocity64 | 3 | depablo_low_temp | teacher_one_step | 0.992 | 1/1 | 20/20 |
| velocity64 | 3 | depablo_low_temp | velocity_one_step | 0.992 | 1/1 | 20/20 |
| velocity64 | 3 | lj_noisy | persistence_one_step | 0.518 | 1/1 | 20/20 |
| velocity64 | 3 | lj_noisy | prefix_prediction | -5.641 | 1/1 | 20/20 |
| velocity64 | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| velocity64 | 3 | lj_noisy | teacher_one_step | 0.520 | 1/1 | 20/20 |
| velocity64 | 3 | lj_noisy | velocity_one_step | 0.521 | 1/1 | 20/20 |
| velocity64 | 3 | reid | persistence_one_step | 0.951 | 1/1 | 20/20 |
| velocity64 | 3 | reid | prefix_prediction | -0.328 | 1/1 | 20/20 |
| velocity64 | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |
| velocity64 | 3 | reid | teacher_one_step | 0.952 | 1/1 | 20/20 |
| velocity64 | 3 | reid | velocity_one_step | 0.952 | 1/1 | 20/20 |
| velocity_node64 | 3 | depablo_low_temp | persistence_one_step | 0.992 | 1/1 | 20/20 |
| velocity_node64 | 3 | depablo_low_temp | prefix_prediction | 0.834 | 1/1 | 20/20 |
| velocity_node64 | 3 | depablo_low_temp | reconstruction | 0.992 | 1/1 | 20/20 |
| velocity_node64 | 3 | depablo_low_temp | teacher_one_step | 0.992 | 1/1 | 20/20 |
| velocity_node64 | 3 | depablo_low_temp | velocity_one_step | 0.992 | 1/1 | 20/20 |
| velocity_node64 | 3 | lj_noisy | persistence_one_step | 0.518 | 1/1 | 20/20 |
| velocity_node64 | 3 | lj_noisy | prefix_prediction | -10.577 | 1/1 | 20/20 |
| velocity_node64 | 3 | lj_noisy | reconstruction | 0.520 | 1/1 | 20/20 |
| velocity_node64 | 3 | lj_noisy | teacher_one_step | 0.521 | 1/1 | 20/20 |
| velocity_node64 | 3 | lj_noisy | velocity_one_step | 0.521 | 1/1 | 20/20 |
| velocity_node64 | 3 | reid | persistence_one_step | 0.951 | 1/1 | 20/20 |
| velocity_node64 | 3 | reid | prefix_prediction | -0.036 | 1/1 | 20/20 |
| velocity_node64 | 3 | reid | reconstruction | 0.952 | 1/1 | 20/20 |
| velocity_node64 | 3 | reid | teacher_one_step | 0.952 | 1/1 | 20/20 |
| velocity_node64 | 3 | reid | velocity_one_step | 0.952 | 1/1 | 20/20 |

## Run outcomes

 seed          family  observed_frames                run_name    status              job_id
  123         state32                3         state32_h3_s123 completed 4680548.zeus-master
  123         state64                3         state64_h3_s123 completed 4680549.zeus-master
  123        state128                3        state128_h3_s123 completed 4680550.zeus-master
  123      velocity64                3      velocity64_h3_s123 completed 4680551.zeus-master
  123     velocity128                3     velocity128_h3_s123 completed 4680552.zeus-master
  123 velocity_node64                3 velocity_node64_h3_s123 completed 4680553.zeus-master
