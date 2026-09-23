# High-initial-response gradient engineering: study_v3

The selected original networks come from existing train/validation trajectories and readout calibration; this is targeted development, not a held-out test. Each physical result verifies a frozen final graph and did not select an optimizer iterate.

| Source | Network | Variant | Kind | q radius (physical displacement <=0.25 q-radius × shortest original bond) | Original physical p | Final physical p | Change | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reid | 65 | original | original |  | 0.387370 | 0.387370 | +0.000000 | completed |
| reid | 65 | positions_r2_b0 | ml | 2 | 0.387370 | 0.049449 | -0.337921 | completed |
| reid | 65 | positions_r4_b0 | ml | 4 | 0.387370 | 0.009108 | -0.378263 | completed |
| reid | 65 | positions_r4_b0_permuted | random | 4 | 0.387370 | 0.351209 | -0.036161 | completed |
| reid | 93 | original | original |  | 0.381989 | 0.381989 | +0.000000 | completed |
| reid | 93 | positions_r2_b0 | ml | 2 | 0.381989 | 0.043504 | -0.338484 | completed |
| reid | 93 | positions_r4_b0 | ml | 4 | 0.381989 | 0.059288 | -0.322701 | completed |
| reid | 93 | positions_r4_b0_permuted | random | 4 | 0.381989 | 0.345795 | -0.036194 | completed |
| depablo_low_temp | 13 | original | original |  | 0.363512 | 0.363512 | +0.000000 | completed |
| depablo_low_temp | 13 | positions_r2_b0 | ml | 2 | 0.363512 | 0.138950 | -0.224562 | completed |
| depablo_low_temp | 13 | positions_r4_b0 | ml | 4 | 0.363512 | 0.077235 | -0.286277 | completed |
| depablo_low_temp | 13 | positions_r4_b0_permuted | random | 4 | 0.363512 | 0.294855 | -0.068657 | completed |
| depablo_low_temp | 50 | original | original |  | 0.369090 | 0.369090 | +0.000000 | completed |
| depablo_low_temp | 50 | positions_r2_b0 | ml | 2 | 0.369090 | 0.102984 | -0.266106 | completed |
| depablo_low_temp | 50 | positions_r4_b0 | ml | 4 | 0.369090 | 0.061260 | -0.307830 | completed |
| depablo_low_temp | 50 | positions_r4_b0_permuted | random | 4 | 0.369090 | 0.300460 | -0.068630 | completed |

The position-permutation controls are geometrically feasible frozen controls only. They are not norm matched and do not support a matched-control comparison.

| Source | Network | Variant | Control q-norm / ML q-norm | Shrink factor |
| --- | --- | --- | --- | --- |
| reid | 65 | positions_r4_b0_permuted | 0.493238 | 0.500000 |
| reid | 93 | positions_r4_b0_permuted | 0.493928 | 0.500000 |
| depablo_low_temp | 13 | positions_r4_b0_permuted | 0.498095 | 0.500000 |
| depablo_low_temp | 50 | positions_r4_b0_permuted | 0.499548 | 0.500000 |

Study v3 was declared after partial v2 physics and is a separate pristine-start, position-only reused-development comparison. It does not merge with v1 or v2.

All rows, including controls and failures, remain in results.csv. Source-wise counts are in source_summary.csv; frozen provenance is in frozen_designs.json, recipe.json, and verification_recipe.json.
