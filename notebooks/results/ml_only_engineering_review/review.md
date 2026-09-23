# ML-only gradient engineering results

Frozen AE gradients produced positive-to-negative responses in Reid53, Reid59 and Reid57, with LAMMPS used only for final verification. The unchanged method improved all four previously unengineered validation networks (2/2 Reid, 2/2 low-T). Strong edits remain unreliable on the positive-p low-T development networks; adding the tested propagator did not resolve that limitation.

All designs start from original networks. Each study freezes every final graph before a separate LAMMPS verification job. No simulator result selects an optimizer iterate. Studies1–3 use six previously studied development networks and one frozen AE seed786. Study4 applies the exact study3 method to four previously unengineered AE-validation networks. Study5 uses two frozen AE+propagator seeds786/123 on the three low-T development networks. None uses the reserved final test.

Studies1–4 use frozen AEs with downstream source-calibrated affine latent readouts. Study5 instead differentiates physical positions from199 autonomous AE+propagator steps, without a response readout. The readout is a directional proxy; its numerical output is not a reliable physical-response prediction under edits.

Studies: v1 target-based ML optimization; v2 zero-mean log-stiffness projection; v3 fixed-budget centered normalized-gradient descent. Study4 is a new-network check of v3; study5 uses a physical-position objective through a frozen propagator. All initial geometry/topology is fixed, positive springs remain positive, zero springs remain zero. All physical checks use120 quasistatic increments and the physical first/last boundary-node ratio.

| Study | Source | Policy | Seed | Bound | Valid/total | Improved | Negative | ≤−0.1 | Beats random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | depablo_low_temp | ridge2d | 786 | 0.7 | 3/3 | 0 | 0 | 0 | 1/3 |
| 1 | depablo_low_temp | ridge2d | 786 | 2 | 3/3 | 0 | 0 | 0 | 1/3 |
| 1 | depablo_low_temp | z1 | 786 | 0.7 | 3/3 | 0 | 0 | 0 | 1/3 |
| 1 | depablo_low_temp | z1 | 786 | 2 | 3/3 | 0 | 0 | 0 | 1/3 |
| 1 | reid | ridge2d | 786 | 0.7 | 3/3 | 1 | 0 | 0 | 3/3 |
| 1 | reid | ridge2d | 786 | 2 | 3/3 | 1 | 0 | 0 | 3/3 |
| 1 | reid | z1 | 786 | 0.7 | 3/3 | 2 | 0 | 0 | 3/3 |
| 1 | reid | z1 | 786 | 2 | 3/3 | 2 | 0 | 0 | 3/3 |
| 2 | depablo_low_temp | ridge2d | 786 | 0.7 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | depablo_low_temp | ridge2d | 786 | 2 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | depablo_low_temp | z1 | 786 | 0.7 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | depablo_low_temp | z1 | 786 | 2 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | reid | ridge2d | 786 | 0.7 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | reid | ridge2d | 786 | 2 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | reid | z1 | 786 | 0.7 | 3/3 | 3 | 0 | 0 | 3/3 |
| 2 | reid | z1 | 786 | 2 | 3/3 | 3 | 0 | 0 | 3/3 |
| 3 | depablo_low_temp | ridge2d | 786 | 2 | 3/3 | 1 | 0 | 0 | 1/3 |
| 3 | depablo_low_temp | z1 | 786 | 2 | 3/3 | 1 | 0 | 0 | 1/3 |
| 3 | reid | ridge2d | 786 | 2 | 3/3 | 3 | 2 | 1 | 3/3 |
| 3 | reid | z1 | 786 | 2 | 3/3 | 3 | 2 | 1 | 3/3 |
| 4 | depablo_low_temp | ridge2d | 786 | 2 | 2/2 | 2 | 2 | 2 | 2/2 |
| 4 | depablo_low_temp | z1 | 786 | 2 | 2/2 | 2 | 2 | 2 | 2/2 |
| 4 | reid | ridge2d | 786 | 2 | 2/2 | 2 | 2 | 2 | 2/2 |
| 4 | reid | z1 | 786 | 2 | 2/2 | 2 | 2 | 2 | 2/2 |
| 5 | depablo_low_temp | rollout | 123 | 2 | 3/3 | 1 | 0 | 0 | 0/3 |
| 5 | depablo_low_temp | rollout | 786 | 2 | 3/3 | 1 | 0 | 0 | 2/3 |

Final fixed-budget and transfer results (every ML design in studies3–5):

| Study | Source | Network | Policy | Seed | Original | Final | Change | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | reid | 2 | z1 | 786 | 0.224348 | 0.080784 | -0.143564 | completed |
| 3 | reid | 2 | ridge2d | 786 | 0.224348 | 0.082567 | -0.141781 | completed |
| 3 | reid | 53 | z1 | 786 | 0.090498 | -0.144627 | -0.235125 | completed |
| 3 | reid | 53 | ridge2d | 786 | 0.090498 | -0.143736 | -0.234234 | completed |
| 3 | reid | 59 | z1 | 786 | 0.163656 | -0.013985 | -0.177642 | completed |
| 3 | reid | 59 | ridge2d | 786 | 0.163656 | -0.004397 | -0.168054 | completed |
| 3 | depablo_low_temp | 70 | z1 | 786 | 0.331294 | 0.350093 | +0.018800 | completed |
| 3 | depablo_low_temp | 70 | ridge2d | 786 | 0.331294 | 0.350096 | +0.018802 | completed |
| 3 | depablo_low_temp | 78 | z1 | 786 | 0.356878 | 0.419343 | +0.062465 | completed |
| 3 | depablo_low_temp | 78 | ridge2d | 786 | 0.356878 | 0.419896 | +0.063018 | completed |
| 3 | depablo_low_temp | 96 | z1 | 786 | 0.189101 | 0.129326 | -0.059774 | completed |
| 3 | depablo_low_temp | 96 | ridge2d | 786 | 0.189101 | 0.129193 | -0.059908 | completed |
| 4 | reid | 57 | z1 | 786 | 0.021784 | -0.101882 | -0.123665 | completed |
| 4 | reid | 57 | ridge2d | 786 | 0.021784 | -0.101973 | -0.123757 | completed |
| 4 | reid | 15 | z1 | 786 | -0.144451 | -0.273594 | -0.129143 | completed |
| 4 | reid | 15 | ridge2d | 786 | -0.144451 | -0.264848 | -0.120397 | completed |
| 4 | depablo_low_temp | 105 | z1 | 786 | -0.603396 | -0.622435 | -0.019039 | completed |
| 4 | depablo_low_temp | 105 | ridge2d | 786 | -0.603396 | -0.623747 | -0.020351 | completed |
| 4 | depablo_low_temp | 152 | z1 | 786 | -0.503947 | -0.553472 | -0.049526 | completed |
| 4 | depablo_low_temp | 152 | ridge2d | 786 | -0.503947 | -0.553833 | -0.049886 | completed |
| 5 | depablo_low_temp | 70 | rollout | 786 | 0.331294 | 0.337168 | +0.005874 | completed |
| 5 | depablo_low_temp | 70 | rollout | 123 | 0.331294 | 0.338100 | +0.006806 | completed |
| 5 | depablo_low_temp | 78 | rollout | 786 | 0.356878 | 0.348261 | -0.008617 | completed |
| 5 | depablo_low_temp | 78 | rollout | 123 | 0.356878 | 0.349890 | -0.006987 | completed |
| 5 | depablo_low_temp | 96 | rollout | 786 | 0.189101 | 0.215351 | +0.026250 | completed |
| 5 | depablo_low_temp | 96 | rollout | 123 | 0.189101 | 0.209799 | +0.020698 | completed |

Negative and target-attainment columns include already-negative originals. Explicit positive-to-negative conversion counts and newly crossed targets are retained in source_summary.csv. Physical-position rollout p-ratio R² on each model own three optimized designs is in rollout_engineered_prediction_metrics.csv; these are post-optimization development results, not an independent final-test cohort.

The first verifier4675255 failed before physics because the frozen endpoint helper lacked the requested compression_steps argument. Retry4675268 changed only the verifier, preserving the complete v1 design manifest. All failures, original jobs, code hashes, per-step ML objectives, finite-difference gradient checks, final graphs, and physical dumps remain in the study directories.

Larger-bound v1/v2 arms sometimes produce identical designs because the surrogate target is reached before the bound. Do not count those as independent replications. Random controls use one seed per network, not a multi-seed confidence estimate. No p-ratio-based AE training or checkpoint selection occurred.
