# AE reconstruction study review

Completed: 40/40.


## Frame-100 validation reconstruction

| Recipe | Source | Seeds | R² mean ± SD | Minimum seed R² | Position MSE | Valid/total minimum |
|---|---|---:|---:|---:|---:|---:|
| dimension4 | depablo_low_temp | 5 | 0.967 ± 0.011 | 0.956 | 3.16e-06 | 20/20 |
| dimension4 | depablo_mixed_temp | 5 | 0.850 ± 0.076 | 0.724 | 4.65e-05 | 20/20 |
| dimension4 | lj_noisy | 5 | 0.092 ± 0.235 | -0.168 | 8.49e-06 | 20/20 |
| dimension4 | reid | 5 | 0.941 ± 0.021 | 0.903 | 7.08e-06 | 20/20 |
| dimension4_response | depablo_low_temp | 5 | 0.968 ± 0.011 | 0.957 | 3.32e-06 | 20/20 |
| dimension4_response | depablo_mixed_temp | 5 | 0.861 ± 0.056 | 0.781 | 4.66e-05 | 20/20 |
| dimension4_response | lj_noisy | 5 | 0.210 ± 0.285 | -0.246 | 8.59e-06 | 20/20 |
| dimension4_response | reid | 5 | 0.870 ± 0.168 | 0.569 | 7.59e-06 | 20/20 |
| equal_source | depablo_low_temp | 5 | 0.950 ± 0.040 | 0.885 | 3.47e-06 | 20/20 |
| equal_source | depablo_mixed_temp | 5 | 0.682 ± 0.145 | 0.548 | 4.9e-05 | 20/20 |
| equal_source | lj_noisy | 5 | -0.202 ± 1.435 | -2.767 | 8.73e-06 | 20/20 |
| equal_source | reid | 5 | 0.884 ± 0.130 | 0.657 | 7.54e-06 | 20/20 |
| lj_edges | depablo_low_temp | 5 | 0.949 ± 0.025 | 0.908 | 3.56e-06 | 20/20 |
| lj_edges | depablo_mixed_temp | 5 | 0.287 ± 0.361 | -0.109 | 3.17e-05 | 20/20 |
| lj_edges | lj_noisy | 5 | -0.165 ± 0.229 | -0.422 | 8.62e-06 | 20/20 |
| lj_edges | reid | 5 | 0.950 ± 0.016 | 0.930 | 6.79e-06 | 20/20 |
| lj_edges_response | depablo_low_temp | 5 | 0.954 ± 0.016 | 0.930 | 3.81e-06 | 20/20 |
| lj_edges_response | depablo_mixed_temp | 5 | 0.375 ± 0.252 | 0.105 | 3.36e-05 | 20/20 |
| lj_edges_response | lj_noisy | 5 | 0.113 ± 0.030 | 0.070 | 8.61e-06 | 20/20 |
| lj_edges_response | reid | 5 | 0.933 ± 0.030 | 0.884 | 7.01e-06 | 20/20 |
| longer | depablo_low_temp | 5 | 0.967 ± 0.014 | 0.950 | 2.87e-06 | 20/20 |
| longer | depablo_mixed_temp | 5 | 0.684 ± 0.124 | 0.532 | 5.4e-05 | 20/20 |
| longer | lj_noisy | 5 | 0.244 ± 0.230 | -0.049 | 8.62e-06 | 20/20 |
| longer | reid | 5 | 0.949 ± 0.011 | 0.934 | 6.96e-06 | 20/20 |
| response_selection | depablo_low_temp | 5 | 0.921 ± 0.091 | 0.760 | 5.21e-06 | 20/20 |
| response_selection | depablo_mixed_temp | 5 | 0.687 ± 0.260 | 0.318 | 4.46e-05 | 20/20 |
| response_selection | lj_noisy | 5 | 0.305 ± 0.488 | -0.538 | 8.84e-06 | 20/20 |
| response_selection | reid | 5 | 0.918 ± 0.098 | 0.743 | 7.69e-06 | 20/20 |
| wider | depablo_low_temp | 5 | 0.954 ± 0.030 | 0.911 | 3.29e-06 | 20/20 |
| wider | depablo_mixed_temp | 5 | 0.403 ± 0.370 | -0.038 | 6.11e-05 | 20/20 |
| wider | lj_noisy | 5 | -0.242 ± 0.818 | -1.565 | 9.02e-06 | 20/20 |
| wider | reid | 5 | 0.841 ± 0.117 | 0.693 | 8.3e-06 | 20/20 |

## Continuation instructions

Read experiment_results_index.md, the latest 06b experiment log, every candidate recipe, sourcewise_results.csv, and retained_source_ranking.csv.
Inspect failures and paired-seed changes relative to the existing mixed-T-unseen AE baseline. Review all horizons and strain errors, not endpoint rank alone.
Mixed-T is evaluation-only and must not select the recipe. Historical response-selection variants violate the current dynamics-only requirement and must not be promoted. Expert observables are diagnostic only.
Do not advance to LJ propagator supervision merely because one recipe ranks first. Use state/trajectory reconstruction for selection; expert response metrics are post-training diagnostics only.
Preserve the successful 2D mixed-T-transfer baseline. Keep shared-node mendels_q placement and exact saved recipes for future runs.

This packet was generated automatically. It does not represent a new assistant review or automatically launch a new training study.
