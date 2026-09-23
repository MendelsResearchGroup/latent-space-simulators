# High-initial-response gradient engineering: joint_v1

The selected original networks come from existing train/validation trajectories and readout calibration; this is targeted development, not a held-out test. Each physical result verifies a frozen final graph and did not select an optimizer iterate.

| Source | Network | Variant | Kind | q radius (physical displacement <=0.25 q-radius × shortest original bond) | Original physical p | Final physical p | Change | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reid | 65 | original | original |  | 0.387370 | 0.387370 | +0.000000 | completed |
| reid | 65 | joint_r4_b0.25 | ml | 4 | 0.387370 | 0.060263 | -0.327107 | completed |
| reid | 65 | joint_r4_b1 | ml | 4 | 0.387370 | 0.129303 | -0.258067 | completed |
| reid | 65 | joint_r4_b1_permuted | random | 4 | 0.387370 | 0.340716 | -0.046654 | completed |
| reid | 93 | original | original |  | 0.381989 | 0.381989 | +0.000000 | completed |
| reid | 93 | joint_r4_b0.25 | ml | 4 | 0.381989 | 0.079099 | -0.302890 | completed |
| reid | 93 | joint_r4_b1 | ml | 4 | 0.381989 | 0.095988 | -0.286001 | completed |
| reid | 93 | joint_r4_b1_permuted | random | 4 | 0.381989 | 0.311122 | -0.070866 | completed |
| depablo_low_temp | 13 | original | original |  | 0.363512 | 0.363512 | +0.000000 | completed |
| depablo_low_temp | 13 | joint_r4_b0.25 | ml | 4 | 0.363512 | 0.073765 | -0.289748 | completed |
| depablo_low_temp | 13 | joint_r4_b1 | ml | 4 | 0.363512 | 0.093734 | -0.269778 | completed |
| depablo_low_temp | 13 | joint_r4_b1_permuted | random | 4 | 0.363512 | 0.211909 | -0.151603 | completed |
| depablo_low_temp | 50 | original | original |  | 0.369090 | 0.369090 | +0.000000 | completed |
| depablo_low_temp | 50 | joint_r4_b0.25 | ml | 4 | 0.369090 | 0.061050 | -0.308040 | completed |
| depablo_low_temp | 50 | joint_r4_b1 | ml | 4 | 0.369090 | 0.041388 | -0.327702 | completed |
| depablo_low_temp | 50 | joint_r4_b1_permuted | random | 4 | 0.369090 | 0.279010 | -0.090080 | completed |

The joint-permutation controls preserve the centered stiffness edit, but their position q-norm can shrink to meet geometry constraints. They are therefore not jointly norm matched and do not support a matched-control comparison.

| Source | Network | Variant | Control q-norm / ML q-norm | Shrink factor |
| --- | --- | --- | --- | --- |
| reid | 65 | joint_r4_b1_permuted | 0.492343 | 0.500000 |
| reid | 93 | joint_r4_b1_permuted | 0.493630 | 0.500000 |
| depablo_low_temp | 13 | joint_r4_b1_permuted | 0.499430 | 0.500000 |
| depablo_low_temp | 50 | joint_r4_b1_permuted | 0.499456 | 0.500000 |

Joint v1 is a separate pristine-start, 1,200-step joint stiffness-and-position development comparison. It reports both predeclared stiffness bounds for every network and does not select a winner after verification.

All rows, including controls and failures, remain in results.csv. Source-wise counts are in source_summary.csv; frozen provenance is in frozen_designs.json, recipe.json, and verification_recipe.json.
