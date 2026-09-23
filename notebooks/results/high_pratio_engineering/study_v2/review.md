# High-initial-response gradient engineering: study_v2

The selected original networks come from existing train/validation trajectories and readout calibration; this is targeted development, not a held-out test. Each physical result verifies a frozen final graph and did not select an optimizer iterate.

| Source | Network | Variant | Kind | q radius (physical displacement <=0.25 q-radius × shortest original bond) | Original physical p | Final physical p | Change | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reid | 65 | original | original |  | 0.387370 | 0.387370 | +0.000000 | completed |
| reid | 65 | positions_r2_b0 | ml | 2 | 0.387370 | 0.070171 | -0.317199 | completed |
| reid | 65 | positions_r4_b0 | ml | 4 | 0.387370 | 0.057378 | -0.329992 | completed |
| reid | 65 | positions_r4_b0_permuted | random | 4 | 0.387370 | 0.359241 | -0.028129 | completed |
| reid | 65 | joint_r4_b0.25 | ml | 4 | 0.387370 | 0.098892 | -0.288478 | completed |
| reid | 93 | original | original |  | 0.381989 | 0.381989 | +0.000000 | completed |
| reid | 93 | positions_r2_b0 | ml | 2 | 0.381989 | 0.069166 | -0.312822 | completed |
| reid | 93 | positions_r4_b0 | ml | 4 | 0.381989 | 0.144866 | -0.237123 | completed |
| reid | 93 | positions_r4_b0_permuted | random | 4 | 0.381989 | 0.361471 | -0.020517 | completed |
| reid | 93 | joint_r4_b0.25 | ml | 4 | 0.381989 | 0.162630 | -0.219359 | completed |
| depablo_low_temp | 13 | original | original |  | 0.363512 | 0.363512 | +0.000000 | completed |
| depablo_low_temp | 13 | positions_r2_b0 | ml | 2 | 0.363512 | 0.152710 | -0.210802 | completed |
| depablo_low_temp | 13 | positions_r4_b0 | ml | 4 | 0.363512 | 0.089129 | -0.274383 | completed |
| depablo_low_temp | 13 | positions_r4_b0_permuted | random | 4 | 0.363512 | 0.317070 | -0.046442 | completed |
| depablo_low_temp | 13 | joint_r4_b0.25 | ml | 4 | 0.363512 | 0.083956 | -0.279556 | completed |
| depablo_low_temp | 50 | original | original |  | 0.369090 | 0.369090 | +0.000000 | completed |
| depablo_low_temp | 50 | positions_r2_b0 | ml | 2 | 0.369090 | 0.123321 | -0.245769 | completed |
| depablo_low_temp | 50 | positions_r4_b0 | ml | 4 | 0.369090 | 0.102166 | -0.266924 | completed |
| depablo_low_temp | 50 | positions_r4_b0_permuted | random | 4 | 0.369090 | 0.319126 | -0.049964 | completed |
| depablo_low_temp | 50 | joint_r4_b0.25 | ml | 4 | 0.369090 | 0.100380 | -0.268710 | completed |

The position-permutation controls are geometrically feasible frozen controls only. They are not norm matched and do not support a matched-control comparison.

| Source | Network | Variant | Control q-norm / ML q-norm | Shrink factor |
| --- | --- | --- | --- | --- |
| reid | 65 | positions_r4_b0_permuted | 0.491756 | 0.500000 |
| reid | 93 | positions_r4_b0_permuted | 0.492132 | 0.500000 |
| depablo_low_temp | 13 | positions_r4_b0_permuted | 0.497807 | 0.500000 |
| depablo_low_temp | 50 | positions_r4_b0_permuted | 0.499383 | 0.500000 |

Study v2 is a separate reused-development-budget comparison after the v1 diagnosis; it does not replace or merge with v1.

All rows, including controls and failures, remain in results.csv. Source-wise counts are in source_summary.csv; frozen provenance is in frozen_designs.json, recipe.json, and verification_recipe.json.
