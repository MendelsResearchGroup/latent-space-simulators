# High-initial-response gradient engineering: study_v1

The selected original networks come from existing train/validation trajectories and readout calibration; this is targeted development, not a held-out test. Each physical result verifies a frozen final graph and did not select an optimizer iterate.

| Source | Network | Variant | Kind | q radius (physical displacement <=0.25 q-radius × shortest original bond) | Original physical p | Final physical p | Change | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reid | 65 | original | original |  | 0.387370 | 0.387370 | +0.000000 | completed |
| reid | 65 | stiffness_b2 | ml |  | 0.387370 | 0.474548 | +0.087178 | completed |
| reid | 65 | stiffness_b2_permuted | random |  | 0.387370 | 0.401057 | +0.013687 | completed |
| reid | 65 | stiffness_b4 | ml |  | 0.387370 | 0.461415 | +0.074045 | completed |
| reid | 65 | stiffness_b4_permuted | random |  | 0.387370 | 0.370709 | -0.016662 | completed |
| reid | 65 | positions_b0 | ml |  | 0.387370 | 0.335775 | -0.051595 | completed |
| reid | 65 | joint_b4 | ml |  | 0.387370 | 0.354053 | -0.033317 | completed |
| reid | 93 | original | original |  | 0.381989 | 0.381989 | +0.000000 | completed |
| reid | 93 | stiffness_b2 | ml |  | 0.381989 | 0.397450 | +0.015461 | completed |
| reid | 93 | stiffness_b2_permuted | random |  | 0.381989 | 0.394776 | +0.012787 | completed |
| reid | 93 | stiffness_b4 | ml |  | 0.381989 | 0.417044 | +0.035055 | completed |
| reid | 93 | stiffness_b4_permuted | random |  | 0.381989 | 0.351434 | -0.030555 | completed |
| reid | 93 | positions_b0 | ml |  | 0.381989 | 0.294475 | -0.087513 | completed |
| reid | 93 | joint_b4 | ml |  | 0.381989 | 0.335351 | -0.046637 | completed |
| depablo_low_temp | 13 | original | original |  | 0.363512 | 0.363512 | +0.000000 | completed |
| depablo_low_temp | 13 | stiffness_b2 | ml |  | 0.363512 | 0.481643 | +0.118131 | completed |
| depablo_low_temp | 13 | stiffness_b2_permuted | random |  | 0.363512 | 0.340911 | -0.022601 | completed |
| depablo_low_temp | 13 | stiffness_b4 | ml |  | 0.363512 | 0.481035 | +0.117523 | completed |
| depablo_low_temp | 13 | stiffness_b4_permuted | random |  | 0.363512 | 0.378848 | +0.015336 | completed |
| depablo_low_temp | 13 | positions_b0 | ml |  | 0.363512 | 0.317483 | -0.046029 | completed |
| depablo_low_temp | 13 | joint_b4 | ml |  | 0.363512 | 0.385857 | +0.022345 | completed |
| depablo_low_temp | 50 | original | original |  | 0.369090 | 0.369090 | +0.000000 | completed |
| depablo_low_temp | 50 | stiffness_b2 | ml |  | 0.369090 | 0.454743 | +0.085653 | completed |
| depablo_low_temp | 50 | stiffness_b2_permuted | random |  | 0.369090 | 0.399959 | +0.030869 | completed |
| depablo_low_temp | 50 | stiffness_b4 | ml |  | 0.369090 | 0.457816 | +0.088726 | completed |
| depablo_low_temp | 50 | stiffness_b4_permuted | random |  | 0.369090 | 0.348207 | -0.020883 | completed |
| depablo_low_temp | 50 | positions_b0 | ml |  | 0.369090 | 0.310853 | -0.058237 | completed |
| depablo_low_temp | 50 | joint_b4 | ml |  | 0.369090 | 0.341684 | -0.027406 | completed |

All rows, including controls and failures, remain in results.csv. Source-wise counts are in source_summary.csv; frozen provenance is in frozen_designs.json, recipe.json, and verification_recipe.json.
