### Completed spatial-decoder comparison — 2026-09-08

All20 full runs4673337–4673356 and collector4673357 finished with PBS exit0. Last run finished12:47:14 Israel; collector finished12:47:35. All completed artifacts, checkpoints, logs and collector tables synchronized from the donor repository;234 files verified by SHA256. Exact sync manifest, final PBS states, reproducible comparison script, paired seed results and recipe audit: `notebooks/results/compact_lj_spatial_decoder/review_20260908/`. Original status snapshot retained as a historical observation.

Matched seeds3456456/123/456/786/2026, full200 frames,16D node reference, shared train30Reid/30low-T/60LJ, validation20/source; required graph-distance2/3 LJ relations and five-channel edges retained. Recipe assertions confirm identical paired data/split hashes and configured budgets except model/cache paths. New code manifest42802bfb7aa67402172555b8116f0c85af617b9f79450e87c4cc80a36a41d88e; baseline code hashes/checkpoint hashes and job mappings in comparison_metadata.json. Coordinate-only worst-source validation selection; mixed-T post-fit only, no final test.

Frame199 coordinate MSE, mean ± sample SD across five seeds. Every row has100/100 valid evaluations for each model (20 unique networks repeated across five seeds). Negative change means lower error; percentage is change of means, not mean paired percentage (both saved in CSV).

| Spatial variant | Source | Baseline MSE ± SD | Spatial MSE ± SD | Change | Improved seeds |
| --- | --- | --- | --- | --- | --- |
| ljonly_spatial_r16_d4 | lj_noisy | 4.08065e-05 ± 6.13e-07 | 4.35226e-05 ± 1.37e-06 | +6.66% | 0/5 |
| spatial_r16_d2 | depablo_mixed_temp | 3.76944e-05 ± 3.39e-06 | 3.24295e-05 ± 1.86e-06 | -13.97% | 5/5 |
| spatial_r16_d2 | depablo_low_temp | 1.44212e-05 ± 2.31e-06 | 1.17685e-05 ± 3.34e-07 | -18.39% | 5/5 |
| spatial_r16_d2 | lj_noisy | 3.99904e-05 ± 1.45e-06 | 3.90931e-05 ± 5.29e-07 | -2.24% | 2/5 |
| spatial_r16_d2 | reid | 3.19803e-05 ± 6.34e-07 | 3.20449e-05 ± 1.22e-06 | +0.20% | 3/5 |
| spatial_r16_d4 | depablo_mixed_temp | 3.78528e-05 ± 5.22e-06 | 3.10283e-05 ± 7.24e-07 | -18.03% | 5/5 |
| spatial_r16_d4 | depablo_low_temp | 1.47619e-05 ± 1.68e-06 | 1.25005e-05 ± 9.76e-07 | -15.32% | 4/5 |
| spatial_r16_d4 | lj_noisy | 3.94145e-05 ± 1.8e-07 | 3.86328e-05 ± 5.74e-07 | -1.98% | 5/5 |
| spatial_r16_d4 | reid | 3.16988e-05 ± 1.45e-06 | 3.07833e-05 ± 1.12e-06 | -2.89% | 4/5 |
| spatial_r16_d8 | depablo_mixed_temp | 3.74803e-05 ± 3.3e-06 | 3.16401e-05 ± 2.23e-06 | -15.58% | 5/5 |
| spatial_r16_d8 | depablo_low_temp | 1.45689e-05 ± 1.02e-06 | 1.21245e-05 ± 8.85e-07 | -16.78% | 5/5 |
| spatial_r16_d8 | lj_noisy | 4.01114e-05 ± 1.14e-06 | 3.89172e-05 ± 1.08e-06 | -2.98% | 4/5 |
| spatial_r16_d8 | reid | 3.14633e-05 ± 4.81e-07 | 3.24581e-05 ± 8.41e-07 | +3.16% | 1/5 |

At frame100 shared LJ coordinate error is2.25/3.07/3.66% higher for D2/4/8; at199 only2.24/1.98/2.98% lower. LJ-only D4 is7.62% worse at100 and6.66% worse at199 (allfive paired seeds worse at199). Late low-T and mixed-T reconstruction improve materially; Reid is small/inconsistent. This architecture has not resolved the coordinate reconstruction limitation. Larger latent dimensions still provide no clear LJ rescue.

Important claim boundary: no p-ratio reconstruction R² was computed in this new sweep. Coordinate MSE does not establish p-ratio failure. Shared spatialD4 LJ mean per-network displacement-explained score is0.8325 at100 and0.8077 at199 relative to zero displacement, so reconstruction is not wholly absent. This score is not p-ratio R². Historical poor p-ratio results concern earlier recipes. No propagator was trained.

Architecture changes parameter count/initialization and realized stopping epochs: configured budgets match, realized updates do not. Shared-vs-LJ-only also differs in fitted normalization/exposure. Allfive LJ-only recipe files incorrectly state shared_fit=true; source.dataset_mixture contains LJ only and training exposure confirms60LJ/12000 frames. Treat this as a metadata flag bug, not shared fitting; preserve original recipes.

Interpretation: decoder self-attention gives useful source-dependent coordinate gains but modest LJ gains. The earlier nonlinear topology-aware reference-encoding hypothesis remains motivated, not established. Post-fit p-ratio evaluation is still needed to answer the response-reconstruction question for these checkpoints; it must not enter fitting or selection.
