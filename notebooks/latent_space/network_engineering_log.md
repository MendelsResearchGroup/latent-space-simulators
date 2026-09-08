# Network engineering log

## Endpoint engineering validation — 2026-09-07

Completed exploratory validation only; final test untouched. Frozen 2D AE job
`4670957.zeus-master` and helper endpoint reproduction `4670994.zeus-master`
succeeded before source jobs `4670995.zeus-master` (Reid) and
`4670996.zeus-master` (dePablo low-T). Each source completed 3/3 selected
validation-design candidates, seed 786. The frozen code manifest is
`notebooks/results/network_design/code_endpoint_v1/hashes.json` with manifest
SHA-256 `943fd615237c67cad58e95073b867c4abeb1fb3d7d89adbe69d5351fa7b85f`.

Protocol: 150 bounded log-stiffness steps (±0.25) using a standardized initial
2D latent affine ridge readout (ridge 0.01); original, designed, opposite,
five matched-random, and uniform-scale variants. Physics reports the first-to-
last unwrapped boundary-node ratio under 2D quasistatic compression to matched
final x scale, y stress zero, fixed shear, with initial/final output only.

| Source | Candidate | Designed minus original | Random controls beaten | Uniform-scale change |
|---|---:|---:|---:|---:|
| Reid | 2 | -0.001548 | 2/5 | +1.66e-08 |
| Reid | 53 | -0.002052 | 3/5 | +1.26e-08 |
| Reid | 59 | -0.003162 | 5/5 | -3.58e-09 |
| dePablo low-T | 70 | -0.002369 | 3/5 | -1.14e-06 |
| dePablo low-T | 78 | +0.000339 | 2/5 | -2.30e-07 |
| dePablo low-T | 96 | +0.009402 | 0/5 | -3.56e-09 |

Reid has 3/3 negative physical changes. dePablo low-T has one negative and two
positive changes. The observed random-control comparisons and one-seed,
validation-subdivision results do not establish reliable engineering.

Machine-readable sources: `notebooks/results/network_design/engineering_endpoint_v1/engineering_summary.json`,
`submission_recipe.json`, `reid/reid_engineering_comparison.csv`, and
`depablo_low_temp/depablo_low_temp_engineering_comparison.csv`. The collector
was `4670998.zeus-master` after-any dependency on both source jobs.

### Source-calibrated engineering launched — 2026-09-07

User clarified that dataset-specific ascending/descending latent order must be
established before editing. On the existing10 calibration graphs/source,
Pearson(initial z0, final p): Reid -0.97599; dePablo low-T +0.98243;
z1: Reid -0.97195; low-T -0.99521. Notebook06 explicitly computes signed
within-dataset correlations before its absolute-correlation summary; its
cross-network ordering motivates this intervention experiment but does not
by itself establish local edit response.

Previous signed-control audit: `engineering_endpoint_v1/terra_direction_audit.json`.
Only low-T96 improved with the previous opposite edit; low-T78 worsened in both
directions. New local-gradient pilot4671363 uses the same frozenAE and source
ridge readout, removes uniform log-stiffness scaling from directions, and tests
±0.01/0.05/0.15/0.25 before four rounds of physics-accepted updates versus equal-
budget random directions. Its autograd vs finite-difference relative error is
0.000170. Early low-T96 endpoint p falls0.189101→0.185308 at +0.25 local step;
this is a probe result, not yet final optimized/control-confirmed performance.
Pilot recipes/evaluations: `network_design/direction_v1/`.

The main follow-up uses four separate calibration graphs/source, chosen at
ranks0/3/6/9 of the10 calibration responses. Compare z0,z1,PC1 and ridge2D,
with ±0.025/0.1 local normalized gradients. Choose one source-level policy/sign
by improvement count then median small-edit change, using calibration only.
A policy is marked reliable only with at least3/4 small-edit improvements.
Apply to the existing disjoint Reid2/53/59 and low-T70/78/96 design networks.
Four arms: calibrated fixed sign, calibrated bidirectional, original ridge
fixed sign, random bidirectional. Each starts unchanged, gets4 proposals×4
rounds, max|log(k/k0)|0.25, accepts only physical decreases>1e-5. Finer60-step
endpoint confirmations compare original and all final candidates (normal30).
556 planned physical evaluations; fixed geometry/topology and zero springs.
One random seed per candidate (786+index); no multi-seed robustness claimed.
AE unchanged, response used only downstream, no final test. ExistingAE was
trained with the101-frame prefix; this held-frozen intervention comparison
does not claim to validate full-trajectory AE reconstruction.

Exact recipes, hashes, split IDs, jobs/dependencies, individual successes and
failures: `network_design/calibrated_direction_v1/{plan.json,jobs.jsonl,code/hashes.json}`.
Shared mendels_q8CPU16GB per job, collector1CPU2GB. Dependent automatic collection
saves source-wise valid/total counts and paired control differences. Inspection:
`notebooks/network_design/02_source_calibrated_engineering.ipynb`.

### Local-gradient pilot completed — 2026-09-07

Validation-design low-T96 pilot `4671363.zeus-master` completed 44/44 physical
evaluations (seed786; final test untouched). The frozen AE/readout local-gradient
check agreed with finite difference (relative error0.000170). Both arms had four
rounds of four physical proposals (16 per arm) and accepted four rounds. Under the
finer 60-step confirmation, AE-gradient changed endpoint p
0.189101→0.183233 (-0.005868); equal-budget bidirectional random changed it
0.189101→0.187210 (-0.001891). The AE arm's reduction exceeded random by0.003977
(3.10× its absolute reduction). This is one candidate and one seed, so it does
not establish reliable engineering. Physics was 2D quasistatic matched endpoint
compression, y stress zero/fixed shear, using the unwrapped first-to-last
boundary-node ratio; response remained downstream only.

Inputs and exact 30-step/60-step values: `network_design/direction_v1/terra_summary.json`,
`depablo_low_temp_96_s786/{completed.json,summary.csv,optimization.csv,gradient_check.json}`.
For context, `old_vs_local_direction.json` finds centered old versus local
gradient cosine -0.1079 and fractions at the edit bound 60.87% versus0.23%; the
directions are not a pure reversal comparison.

### Additional stochastic controls submitted — 2026-09-07

Twelve random-only repeats4671470–81 add seeds123+index and456+index to the
main786+index control for each of the six design networks. Each gets the same
four rounds, four proposals/round, ±0.25 log-stiffness bound and physical
acceptance rule as the main random arm;19 first/last physics evaluations
include original and final60-step confirmation. No deterministic AE arm is
repeated. Shared mendels_q8CPU8GB, justified by the same prepared workload
whose preparation peaked below3GB. Main architecture/training allocations
remain8CPU16GB. Collector4671482 depends on all12 repeats and six main
optimization jobs, saving source-wise valid/total counts and per-network
comparisons against all three random seeds. Artifacts and exact jobs:
`network_design/random_direction_replicates_v1/`. No AE-seed robustness or
untouched test-set conclusion is implied by repeating random controls.

### Joint latent direction audit and inverse-Jacobian test — 2026-09-07

Terra's frozen local sensitivity audit covers4 calibration+3 design graphs per
source. Across calibration graphs z0/z1 correlate+0.996 for Reid and-0.990 for
low-T. Centered stiffness-Jacobian rows correlate positively in both sources.
The old ridge-gradient latent movement follows Reid's PC1 ordering (cos0.979–1),
but is almost perpendicular/opposite for low-T (cos-0.147–0.021). This does not
show impossibility of moving along that ordering: both latent Jacobians are
full rank with modest condition numbers (Reid3.20–6.29, low-T2.32–4.04).
Audit data/code/hashes: `network_design/calibrated_direction_v1/latent_direction_audit.{json,md,py}`.

New targeted test: choose the signed PC1 direction from the calibration graphs,
then solve the centered damped inverse Jacobian for a minimum-norm spring edit
that moves BOTH standardized latent coordinates along that direction. Normalize
maximum log-step, four rounds/four positive proposals each, same0.25 total bound,
physical acceptance>1e-5, and60-step confirmation. Save linearized and actual
latent alignment for every proposal, plus graphs and physical results. Six
original design networks,114 endpoint simulations, same frozen AE; labels only
in downstream calibration/edit acceptance. Compare against calibrated scalar
z1 and the existing3 random seeds; no deterministic baseline duplication.
Exact recipe/jobs/code hashes: `network_design/tangent_direction_v1/`.
Shared mendels_q8CPU8GB; automatic source-wise collector1CPU2GB depends on all
six jobs plus completed prior random/control aggregation. Validation only.

### Source-calibrated engineering collection

Completed 6/6 design networks; failed 0, missing 0. Source-wise metrics and paired random comparisons: `network_design/calibrated_direction_v1/source_results.csv`, `network_results.csv`, `paired_vs_random.csv`. Four independent calibration graphs/source select a common coordinate and sign; all design labels are disjoint from calibration. Physics-accepted optimization and 60-increment confirmation remain exploratory validation, not final test.

### Engineering random-control replication collected

Additional random-seed jobs completed 12/12. Combined with the original random arm, each design network has up to3 equal-budget random optimizers. Source-wise valid/total counts and paired comparisons: `network_design/random_direction_replicates_v1/source_comparisons.csv`, `paired_comparisons.csv`. Validation only; three random seeds do not establish AE seed robustness.

### Joint-latent-direction engineering collected

Completed 6/6 inverse-Jacobian PC1 design jobs. Source-wise counts and comparisons: `network_design/tangent_direction_v1/source_results.csv`, `network_results.csv`, `paired_comparison.csv`. Same frozen AE, source-calibration PC1 orientation, four rounds/four proposals, first-last physical acceptance and60-increment confirmation. Validation only.

### Best physically verified designs exported

Six original/engineered graph pairs and LAMMPS network inputs: `network_design/best_verified_designs_v1/`. Geometry/topology unchanged, zero springs preserved, |log(k/k0)|<=0.25 checked. Each winner selected among seven confirmed method/seed candidates; this is engineering selection, not an unbiased method comparison. Exact recipe/hash/protocol per-network provenance.json; source/network results summary.csv.

### Engineering conclusion after source calibration and controls — 2026-09-07

All experiment chains and collectors completed:8 calibration jobs136/136 physical
checks; six main jobs420/420 checks;12 extra random jobs228/228; six inverse-
Jacobian jobs114/114; earlier pilot44/44. Total942 completed endpoint simulations.
Rechecked counts from per-run artifacts below. Frozen AE has not changed.

The simple source-calibrated z1 fixed-sign optimizer improved6/6 design networks.
The bidirectional version found identical final graphs/responses for these six;
no per-network sign reversal was needed here. All reductions below use60-step
confirmation and fresh unchanged baselines, after four accepted local updates:

| Source | Index | Original | Calibrated z1 | Change | Reduction |
|---|---:|---:|---:|---:|---:|
| depablo_low_temp | 70 | 0.331294 | 0.328653 | -0.002640 | 0.80% |
| depablo_low_temp | 78 | 0.356878 | 0.355500 | -0.001377 | 0.39% |
| depablo_low_temp | 96 | 0.189101 | 0.183553 | -0.005548 | 2.93% |
| reid | 2 | 0.224348 | 0.204250 | -0.020098 | 8.96% |
| reid | 53 | 0.090498 | 0.051186 | -0.039312 | 43.44% |
| reid | 59 | 0.163656 | 0.132941 | -0.030716 | 18.77% |

Against three equal-budget random seeds/network, calibrated z1 beats the random
mean on3/3 Reid and2/3 low-T; it beats all three controls on all Reid graphs and
low-T96. Low-T70 beats2/3 controls; low-T78 loses to3/3 controls. Thus physical
engineering succeeds on these six, with stronger evidence of useful AE guidance
for Reid than uniformly across low-T. This is one frozenAE seed and six selected
validation-design graphs, not final-test generalization evidence.

Inverse-Jacobian PC1 follows the joint latent ordering accurately, yet improves
only1/3 low-T (70);78/96 accept no changes. It improves3/3 Reid, with slightly
better source mean(-0.031371 vs calibrated-0.030042), but does not justify replacing
the simpler scalar method across both sources. Exact paired counts and failures
are retained, including these negative results. The source-order correlation
is useful but does not by itself specify every physically beneficial stiffness
intervention. Original validation graphs already have heterogeneous stiffnesses,
so that property alone cannot explain the direction mismatch; scope is14 prepared
validation refs, not the full training distribution.

Deliverable: `network_design/best_verified_designs_v1/` contains six best confirmed
original/engineered graph pairs and LAMMPS network inputs, plus per-network recipe,
hashes and method provenance. Export4671574 completed with six successful graph
checks (geometry/topology unchanged, positive/zero stiffness constraints retained).
Those winners include random controls and should not be used as an AE-only
performance table. Main calibrated results and random comparisons remain separate.
Notebook `notebooks/network_design/02_source_calibrated_engineering.ipynb` runs
with the collected results, plots inline and exports no figures.

### Targeted continuation to negative p-ratio — 2026-09-07

User requested engineering one network until negative p-ratio. Choose Reid53,
closest verified result: original0.09049822, best tangent design0.04956001.
Continue the exact saved log-stiffness edit with the same frozenAE. At each
round compare inverse-Jacobian PC1 and calibrated z1 directions at steps0.05/
0.1/0.2; accept only physical improvements>1e-5. The old0.25 experiment bound
is expanded progressively min(1.5,0.4+0.15*round), relative to original stiffness,
to permit crossing zero. Positions/topology stay fixed; positive springs and
zeros preserved. Seed839 random bidirectional fallback only on a stall;24-round
initial budget. Target≤-0.005 must pass both60 and120 compression increments,
with the same first-last unwrapped response and x compression as before.
Frozen code/manifest: `network_design/negative_reid53_v1_code/`; evaluations,
accepted edits, failed attempts, final graph and confirmation dumps:
`network_design/negative_reid53_v1/`. Shared mendels_q8CPU8GB, no host pin.
This is targeted downstream engineering on a validation graph, not AE training
or a generalization benchmark. Target attainment awaits physical results.

### Negative p-ratio achieved on Reid53 — 2026-09-07

PBS4671579 completed in79.05s, maxRSS722592kB,22/22 physical evaluations valid.
Original first/last p=+0.09049821990; prior optimized design=+0.04956001489.
Three accepted continuation rounds: tangent→0.0291421 at bound0.4, calibrated
z1→0.0102586 at bound0.55, calibrated z1→-0.00601217538 at bound0.7.
Negative endpoint response confirmed at60 increments=-0.006012175382014793 and
120 increments=-0.0060121753920150895. Same2D quasistatic x compression scale
0.9700107, y stresszero, fixed shear, first/last unwrapped boundary-node response.
The box-based diagnostic is also negative (about-0.010628). Positions/topology
unchanged; all positive stiffnesses remain positive; achieved multipliers relative
to original0.546115–2.013753. No random fallback was needed. AE frozen; source
calibration and physical acceptance only downstream, no final test or generalized
negative-response guarantee. Stop on target attainment; no further optimization.

Artifacts: `network_design/negative_reid53_v1/{summary.json,recipe.json,optimization.csv,
evaluations.csv,engineered.pt,engineered_network.lmp,original.pt,confirmed_initial.dump,
confirmed_final.dump}`. Original registry-response metadata cleared on edited graph.
Exact script and parent snapshot hashes saved; graph SHA256
2bf3d8e5755d79d5ae6c354411dae6c845030f2257259bf8768a8a31be46960f.
The engineering notebook includes the continuation and confirmation table.

### Node-position and stiffness engineering to p <= -0.1 — 2026-09-07

User explicitly requested position edits, multiple networks, target-0.1 and fast
execution. Prior evidence: source-calibrated z1 improved6/6 with 5/6 beating random
mean; inverse-Jacobian joint latent direction helped Reid but failed two low-T
networks. Reid53 now has confirmed p=-0.006012 after stiffness continuation.

New comparison: Reid2/53/59 and low-T70/78/96 × stiffness-only / positions-only /
joint edits. Every mode starts from the same best previous stiffness graph per
network; Reid53 uses the negative design. This is a continuation comparison,
not a from-scratch or untouched-test benchmark. Frozen AE/normalizers/readouts.
Each step tests z1 and inverse-Jacobian PC1 at step0.1/0.2/0.4; accept measured
physical decreases>1e-5. On a stall, four seeded random directions with both signs.
Up to40 rounds; after six consecutive stalls terminate with best design and an
explicit unmet-target result. Confirm target p<=-0.1 with60 and120 increments.

Geometry edits update physical reference positions, physical minimum-image
vectors/lengths, normalized positions/vectors/lengths together. Zero displacement
features at the new reference frame are retained. MetaForge write_network rebuilds
spring rest lengths from current geometry; no topology reuse. Box, topology,
boundary measurement nodes and side memberships fixed; positive stiffnesses and
zeros retained. Total |log(k/k_original)|<=2, maximum node movement<=half its
original shortest incident bond. Reject bond lengths outside[0.25,1.75] original
and node separations below0.1 original minimum. These are downstream engineering
constraints; no expert-observable AE training or selection.

Geometry identity and canonicalize→normalize→encode round-trip checks passed on
Reid53/low-T96. Position and stiffness gradients agree with finite differences
(relative errors0.00330/0.00472 Reid;0.000533/0.0000416 low-T). Raw physical edge
order must be canonicalized before comparing AE inputs. Checks:
`network_design/check_geometry_contract.py`, results `network_design/geometry_contract_verification.json`.

Speed: one independent shared mendels_q job per network/mode,8CPU8GB4h, no pins.
Four simultaneous candidate simulations, each OMP/BLAS/MKL1; parent Torch4threads.
Each candidate uses its own graph clone/output directory. Cache exact repeated
edits within a run; retain actual simulation and proposal/rejection counts.
Measured earlier optimizer RSS about0.7GiB (separate LAMMPS process not included),
so keep8GB until concurrent workload measured. Two joint smoke jobs first.

Root owns geometry/model reasoning; Terra owns PBS submission/monitoring/collector.
Snapshots, exact recipes/splits/hashes/jobs, failures and per-source results:
`network_design/geometry_target_v1/`; smoke attempts `network_design/geometry_smoke_v1/`.
Notebook `notebooks/network_design/03_geometry_and_stiffness_design.ipynb` displays
results inline without figure exports. No claim of reaching-0.1 yet.

### Geometry matrix submitted; first -0.1 confirmations — 2026-09-07

Both joint2-round smokes passed: Reid53 (4672359) start-0.00601218→-0.06083078,
low-T96 (4672360) +0.18323287→+0.13699384, each15/15 valid and120-step confirmed.
Frozen geometry_target_v1/code manifest SHA256:
84b6a3930367b51c213230ed7ea9798e2ab2c9ddc203473303b2edf6d57242e8.
Full18 unique jobs submitted with recipe saved before launch and snapshot hashes
in each ledger row. Job IDs by source/index, stiffness/positions/joint:
Reid2:4672362/63/64; Reid53:4672365/66/67; Reid59:4672368/69/71;
low-T70:4672372/73/75; low-T78:4672376/78/79; low-T96:4672380/81/82.
All .zeus-master, mendels_q8CPU8GB4h shared without host pins.
Afterany collector4672383 depends on all18 (1CPU2GB). Exact job mapping,
configuration and hashes: geometry_target_v1/jobs.jsonl and submission_recipe.json.

Early terminal successes: Reid53 stiffness4672365 reached120-step p=-0.10344072336
in5 rounds (34/34 valid); Reid53 joint4672367 reached-0.10232924983 in4 rounds
(28/28 valid). Both exited0 with no geometry rejects. These stop at target;
therefore their final p values are not an equal-budget method ranking. Other runs
remain in progress at this entry; source-level success rates await collection.
