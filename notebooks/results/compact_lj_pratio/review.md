# Frozen AE p-ratio results — 2026-09-08

All70 existing checkpoints evaluated successfully:50 compact reconstruction controls and20 spatial-decoder models. Jobs4674041–4674110 all PBS exit0; no training reruns. Primary metric is post-fit physical-coordinate endpoint directional-side p-ratio R², with exact original reference positions defining side groups. Historical normalized-coordinate endpoint scores are retained separately. All23,500/23,500 network-frame evaluations are finite. No final test or propagator evaluation.

Each cell below uses five seeds (3456456,123,456,786,2026) on the same20 validation networks/source:100/100 valid evaluations per horizon, not100 unique networks. Values are seed mean ± sample SD. Mixed-T is unseen in fitting/selection and evaluated separately as reconstruction transfer. All checkpoints were selected with coordinate reconstruction; p-ratio did not enter training or selection.

## LJ reconstruction

| Variant | Frame100 R² | Frame199 R² |
| --- | --- | --- |
| ljonly_r16_d4 | -0.105 ± 0.323 | -1.262 ± 0.870 |
| ljonly_r16_d8 | -0.247 ± 0.476 | -2.944 ± 2.659 |
| ljonly_spatial_r16_d4 | -1.738 ± 0.654 | -3.396 ± 2.602 |
| mp2_r16_d4 | 0.633 ± 0.095 | 0.127 ± 0.284 |
| mp2_r16_d8 | 0.523 ± 0.115 | -0.081 ± 0.304 |
| r16_d2 | 0.451 ± 0.020 | 0.413 ± 0.269 |
| r16_d4 | 0.335 ± 0.112 | 0.360 ± 0.074 |
| r16_d6 | 0.444 ± 0.217 | 0.388 ± 0.103 |
| r16_d8 | 0.332 ± 0.220 | 0.317 ± 0.428 |
| r96_d4 | 0.030 ± 0.225 | 0.389 ± 0.127 |
| r96_d8 | 0.189 ± 0.141 | 0.457 ± 0.195 |
| spatial_r16_d2 | 0.212 ± 0.144 | -0.040 ± 0.133 |
| spatial_r16_d4 | 0.206 ± 0.286 | 0.315 ± 0.470 |
| spatial_r16_d8 | 0.437 ± 0.317 | 0.406 ± 0.254 |

## Interpretation

- Coordinate errors alone gave an overly pessimistic picture. Shared AEs preserve meaningful LJ p-ratio information; they are not total failures.
- MP2 with a4D latent has the highest mean LJ R² at frame100 in this matrix:0.633±0.095. It falls to0.127±0.284 at199. This supports useful mid-trajectory reconstruction, not robust full-horizon performance.
- Standard shared2D achieves0.451±0.020 at100 and0.413±0.269 at199. The 2D/4D/6D/8D results do not show monotonic gains with latent width.
- Wider96D-reference/8D-latent control has frame199 mean0.457±0.195; compact2D and spatial8D remain around0.41 with substantial seed variation. Do not promote a single late-horizon winner as established.
- Spatial decoding reduces LJ coordinate error slightly at199, but worsens2D p-ratio R² at199 by0.453±0.248 paired across seeds (allfive worse). Its4D change is-0.045±0.472; its8D change+0.090±0.581. Coordinate improvements do not guarantee response improvements.
- LJ-only training is weak: D4 endpoint means-0.105/-1.262 at100/199; spatial LJ-only-1.738/-3.396. Shared/LJ-only regimes differ in normalization/exposure, so this is descriptive rather than an isolated interference claim.
- Other sources have strong response reconstruction. Standard shared2D Reid0.931→0.956, low-T0.932→0.945, mixed-T0.840→0.898 (frames100→199). MP2D4 low-T0.988→0.991, Reid0.950→0.924, mixed-T0.884→0.923. All are reconstruction results, not autonomous rollout.

## Source-wise endpoint results

| Variant | Source | Frame | R² mean ± SD | Valid/total |
| --- | --- | --- | --- | --- |
| ljonly_r16_d4 | lj_noisy | 100 | -0.1048 ± 0.3229 | 100/100 |
| ljonly_r16_d4 | lj_noisy | 199 | -1.2623 ± 0.8696 | 100/100 |
| ljonly_r16_d8 | lj_noisy | 100 | -0.2474 ± 0.4763 | 100/100 |
| ljonly_r16_d8 | lj_noisy | 199 | -2.9437 ± 2.6592 | 100/100 |
| ljonly_spatial_r16_d4 | lj_noisy | 100 | -1.7377 ± 0.6536 | 100/100 |
| ljonly_spatial_r16_d4 | lj_noisy | 199 | -3.3964 ± 2.6018 | 100/100 |
| mp2_r16_d4 | depablo_low_temp | 100 | 0.9877 ± 0.0052 | 100/100 |
| mp2_r16_d4 | depablo_low_temp | 199 | 0.9906 ± 0.0009 | 100/100 |
| mp2_r16_d4 | depablo_mixed_temp | 100 | 0.8838 ± 0.0611 | 100/100 |
| mp2_r16_d4 | depablo_mixed_temp | 199 | 0.9229 ± 0.0267 | 100/100 |
| mp2_r16_d4 | lj_noisy | 100 | 0.6330 ± 0.0946 | 100/100 |
| mp2_r16_d4 | lj_noisy | 199 | 0.1270 ± 0.2840 | 100/100 |
| mp2_r16_d4 | reid | 100 | 0.9505 ± 0.0283 | 100/100 |
| mp2_r16_d4 | reid | 199 | 0.9235 ± 0.0458 | 100/100 |
| mp2_r16_d8 | depablo_low_temp | 100 | 0.9855 ± 0.0042 | 100/100 |
| mp2_r16_d8 | depablo_low_temp | 199 | 0.9807 ± 0.0075 | 100/100 |
| mp2_r16_d8 | depablo_mixed_temp | 100 | 0.8779 ± 0.0027 | 100/100 |
| mp2_r16_d8 | depablo_mixed_temp | 199 | 0.8896 ± 0.0181 | 100/100 |
| mp2_r16_d8 | lj_noisy | 100 | 0.5226 ± 0.1149 | 100/100 |
| mp2_r16_d8 | lj_noisy | 199 | -0.0813 ± 0.3042 | 100/100 |
| mp2_r16_d8 | reid | 100 | 0.9635 ± 0.0143 | 100/100 |
| mp2_r16_d8 | reid | 199 | 0.9369 ± 0.0244 | 100/100 |
| r16_d2 | depablo_low_temp | 100 | 0.9322 ± 0.0116 | 100/100 |
| r16_d2 | depablo_low_temp | 199 | 0.9454 ± 0.0120 | 100/100 |
| r16_d2 | depablo_mixed_temp | 100 | 0.8403 ± 0.0491 | 100/100 |
| r16_d2 | depablo_mixed_temp | 199 | 0.8982 ± 0.0244 | 100/100 |
| r16_d2 | lj_noisy | 100 | 0.4508 ± 0.0201 | 100/100 |
| r16_d2 | lj_noisy | 199 | 0.4128 ± 0.2687 | 100/100 |
| r16_d2 | reid | 100 | 0.9311 ± 0.0275 | 100/100 |
| r16_d2 | reid | 199 | 0.9558 ± 0.0206 | 100/100 |
| r16_d4 | depablo_low_temp | 100 | 0.9341 ± 0.0094 | 100/100 |
| r16_d4 | depablo_low_temp | 199 | 0.9535 ± 0.0122 | 100/100 |
| r16_d4 | depablo_mixed_temp | 100 | 0.8607 ± 0.0571 | 100/100 |
| r16_d4 | depablo_mixed_temp | 199 | 0.9193 ± 0.0313 | 100/100 |
| r16_d4 | lj_noisy | 100 | 0.3349 ± 0.1121 | 100/100 |
| r16_d4 | lj_noisy | 199 | 0.3600 ± 0.0740 | 100/100 |
| r16_d4 | reid | 100 | 0.9299 ± 0.0093 | 100/100 |
| r16_d4 | reid | 199 | 0.9466 ± 0.0315 | 100/100 |
| r16_d6 | depablo_low_temp | 100 | 0.9454 ± 0.0169 | 100/100 |
| r16_d6 | depablo_low_temp | 199 | 0.9736 ± 0.0105 | 100/100 |
| r16_d6 | depablo_mixed_temp | 100 | 0.8863 ± 0.0565 | 100/100 |
| r16_d6 | depablo_mixed_temp | 199 | 0.9438 ± 0.0134 | 100/100 |
| r16_d6 | lj_noisy | 100 | 0.4436 ± 0.2172 | 100/100 |
| r16_d6 | lj_noisy | 199 | 0.3879 ± 0.1027 | 100/100 |
| r16_d6 | reid | 100 | 0.9452 ± 0.0144 | 100/100 |
| r16_d6 | reid | 199 | 0.9450 ± 0.0173 | 100/100 |
| r16_d8 | depablo_low_temp | 100 | 0.9379 ± 0.0082 | 100/100 |
| r16_d8 | depablo_low_temp | 199 | 0.9532 ± 0.0287 | 100/100 |
| r16_d8 | depablo_mixed_temp | 100 | 0.8644 ± 0.0642 | 100/100 |
| r16_d8 | depablo_mixed_temp | 199 | 0.9121 ± 0.0474 | 100/100 |
| r16_d8 | lj_noisy | 100 | 0.3319 ± 0.2196 | 100/100 |
| r16_d8 | lj_noisy | 199 | 0.3166 ± 0.4280 | 100/100 |
| r16_d8 | reid | 100 | 0.9351 ± 0.0294 | 100/100 |
| r16_d8 | reid | 199 | 0.9504 ± 0.0338 | 100/100 |
| r96_d4 | depablo_low_temp | 100 | 0.9281 ± 0.0125 | 100/100 |
| r96_d4 | depablo_low_temp | 199 | 0.9540 ± 0.0196 | 100/100 |
| r96_d4 | depablo_mixed_temp | 100 | 0.7785 ± 0.1312 | 100/100 |
| r96_d4 | depablo_mixed_temp | 199 | 0.8643 ± 0.1117 | 100/100 |
| r96_d4 | lj_noisy | 100 | 0.0299 ± 0.2252 | 100/100 |
| r96_d4 | lj_noisy | 199 | 0.3894 ± 0.1269 | 100/100 |
| r96_d4 | reid | 100 | 0.9243 ± 0.0187 | 100/100 |
| r96_d4 | reid | 199 | 0.9483 ± 0.0247 | 100/100 |
| r96_d8 | depablo_low_temp | 100 | 0.9273 ± 0.0156 | 100/100 |
| r96_d8 | depablo_low_temp | 199 | 0.9411 ± 0.0245 | 100/100 |
| r96_d8 | depablo_mixed_temp | 100 | 0.8362 ± 0.0433 | 100/100 |
| r96_d8 | depablo_mixed_temp | 199 | 0.8963 ± 0.0214 | 100/100 |
| r96_d8 | lj_noisy | 100 | 0.1892 ± 0.1415 | 100/100 |
| r96_d8 | lj_noisy | 199 | 0.4573 ± 0.1946 | 100/100 |
| r96_d8 | reid | 100 | 0.9171 ± 0.0243 | 100/100 |
| r96_d8 | reid | 199 | 0.9720 ± 0.0092 | 100/100 |
| spatial_r16_d2 | depablo_low_temp | 100 | 0.9642 ± 0.0215 | 100/100 |
| spatial_r16_d2 | depablo_low_temp | 199 | 0.9661 ± 0.0170 | 100/100 |
| spatial_r16_d2 | depablo_mixed_temp | 100 | 0.9113 ± 0.0170 | 100/100 |
| spatial_r16_d2 | depablo_mixed_temp | 199 | 0.8615 ± 0.0744 | 100/100 |
| spatial_r16_d2 | lj_noisy | 100 | 0.2120 ± 0.1443 | 100/100 |
| spatial_r16_d2 | lj_noisy | 199 | -0.0405 ± 0.1328 | 100/100 |
| spatial_r16_d2 | reid | 100 | 0.8878 ± 0.0288 | 100/100 |
| spatial_r16_d2 | reid | 199 | 0.9316 ± 0.0359 | 100/100 |
| spatial_r16_d4 | depablo_low_temp | 100 | 0.9785 ± 0.0112 | 100/100 |
| spatial_r16_d4 | depablo_low_temp | 199 | 0.9814 ± 0.0097 | 100/100 |
| spatial_r16_d4 | depablo_mixed_temp | 100 | 0.9161 ± 0.0245 | 100/100 |
| spatial_r16_d4 | depablo_mixed_temp | 199 | 0.9186 ± 0.0240 | 100/100 |
| spatial_r16_d4 | lj_noisy | 100 | 0.2063 ± 0.2864 | 100/100 |
| spatial_r16_d4 | lj_noisy | 199 | 0.3149 ± 0.4703 | 100/100 |
| spatial_r16_d4 | reid | 100 | 0.8878 ± 0.0813 | 100/100 |
| spatial_r16_d4 | reid | 199 | 0.9101 ± 0.0734 | 100/100 |
| spatial_r16_d8 | depablo_low_temp | 100 | 0.9755 ± 0.0044 | 100/100 |
| spatial_r16_d8 | depablo_low_temp | 199 | 0.9656 ± 0.0182 | 100/100 |
| spatial_r16_d8 | depablo_mixed_temp | 100 | 0.9313 ± 0.0285 | 100/100 |
| spatial_r16_d8 | depablo_mixed_temp | 199 | 0.9005 ± 0.0680 | 100/100 |
| spatial_r16_d8 | lj_noisy | 100 | 0.4368 ± 0.3165 | 100/100 |
| spatial_r16_d8 | lj_noisy | 199 | 0.4062 ± 0.2536 | 100/100 |
| spatial_r16_d8 | reid | 100 | 0.9075 ± 0.0261 | 100/100 |
| spatial_r16_d8 | reid | 199 | 0.9072 ± 0.0728 | 100/100 |

## Provenance and checks

Full source/frame/per-seed tables, raw per-network predicted/true values, physical/model coordinate conventions, near-zero flags, and paired spatial contrasts: collection/. Exact input recipes, checkpoint hashes, source snapshots and job identities: intended_matrix.json, jobs.jsonl, runs/*/submission_recipe.json, runs/*/recipe.json. Runner SHA25665b2da9aa5e34df1d56b05a8e717696b11e3667974fb50275514485fc7daef7f. Source and metric-module hashes recorded per run. Allfour raw dataset hashes matched their training recipes.

Maximum coordinate-MSE difference from saved training-study evaluation rows8.73e-11 across common frames25/50/100/199. Frame150 is newly evaluated. Maximum physical truth p-ratio difference from independently loaded raw data5.32e-6. Checkpoint hashes unchanged before/after all70 evaluations. Exact-cohort collector checks passed; all235 source/frame aggregate cells contain five seeds. Raw target and synthetic/minimal-graph inverse checks: raw_target_audit/.

Preserved provisional smoke failures and fixes are recorded in the experiment index and original smoke snapshots. They are evaluator/launcher failures, not retraining failures, and are excluded from final results. Frozen source copies remain local; per-file SHA manifests and runner snapshots preserve provenance.
