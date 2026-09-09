# Network engineering log

## Engineering goal clarified — 2026-09-08

User requires gradients through the frozen ML model to guide edits to an original network, preferably using the AE alone, with a learned propagator if needed. LAMMPS must only verify final frozen designs and must not influence optimization. Recorded in AGENTS.md, CONTRIBUTING.md, README.md and `docs/research/engineering_goal.md`. Historical simulator-guided negative-Reid53 results remain valid physical demonstrations but do not meet this goal.

Read-only code/recipe audit confirms the earlier `engineering_endpoint_v1` baseline used 150 ML-only Adam updates with downstream initial-latent ridge readout, learning rate0.02, ±0.25 log-stiffness bound, frozen AE seed786. Frozen manifest943fd615237c67cad58e95073b867c4abeb1fb3d7d89adbe69d5351fa7b85f. The existing final physical checks improved3/3 Reid and1/3 low-T graphs; all6/6 were valid and0/6 attained negative p-ratio. Exact source/index changes and controls are in the original entry below. These are reused historical results, not a new experiment or a blind new test of already inspected networks. No new jobs or duplicate evaluations launched for this protocol clarification.

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

### Verified engineering animations recovered — 2026-09-08

User requested a notebook showing original and engineered negative-p-ratio
animations. Recovered the selected Reid53 stiffness-only design from donor
geometry_target_v1/reid_53_stiffness, preserving its summary, recipe, edits and
graphs. The original is pristine raw Reid53 frame0; the earlier negative design
was only the continuation start and is not substituted for the original.
Original/engineered graph SHA256: c8b8a0d4331e5608fba9a8a8c183f513c64e4353560596a0ce87f3721d45af10 /
b410ac79404d44b561989e5dee854c1d28ed999b6ab713bd268224c79bc4d90e.

Fresh physical replays PBS4674995(original) and4674996(engineered), shared
mendels_q1CPU2GB, write121 actual relaxed states each: initial state plus120
quasistatic compression increments, x-scale0.9700107, transverse y stresszero,
fixed shear,2D enforcement. No animation-frame interpolation or optimization.
Original endpoint p=+0.09049821987696514 (prior+0.09049821989813585; error2.12e-11);
engineered p=-0.10344072336031537 (exact prior match). Same first-to-current
unwrapped boundary-node estimator;120/120 finite post-initial values for each
variant, initial p undefined. Geometry, topology and zero-stiffness pattern
unchanged; positive springs remain positive. Graph140nodes/401stored edges.

Portable payload and provenance: notebooks/results/engineering_animation/reid53/
{trajectories.npz,metadata.json,provenance_manifest.json}; payload SHA256
0329991c55569d5184358eb4f2066894e4be33439dd06d6e327d1d7c5b832df9.
Runner SHA2563d80248b982fe7c4e5fe6cdd4dff96f8b0ff7a5290ea4fcdd4ecee2fe8e0d252;
external simulator/helper hashes and exact scripts/input/dumps retained. Initial
PBS4674980/4674981 failed before LAMMPS due interpreter dependencies and remain
recorded. No AE training or final-test evaluation occurred; this is a targeted,
physically accepted engineering example, not generalization or method-ranking
evidence. Notebook: notebooks/network_design/01_verified_negative_pratio.ipynb.

Engineering notebook execution verified: two embedded playable HTML animations
and one inline response PNG; nine notebooks/74 code cells and eight data files
passed the project checker. Each animation displays every second saved physical
state at actual displacement scale, with the final state included; all121 states
remain in the portable payload. Notebook rendering uses reduced display DPI to
keep the self-contained file practical (~21.6MB); physical data are unchanged.

### ML-only gradient engineering matrix — 2026-09-08

User explicitly requested new engineering and continued diagnosis if unsuccessful. New study `notebooks/results/ml_only_engineering_v1/` starts from pristine Reid2/53/59 and low-T70/78/96, never simulator-selected designs. These are previously examined validation-development graphs, not an untouched test. Frozen AE seed786 and existing source readouts use original calibration responses downstream only; no AE training, checkpoint selection, or simulator-selected policy/sign is introduced.

Hypothesis: differentiate the current source-calibrated latent readout locally throughout optimization, with greater allowed stiffness changes than the earlier ±0.25 baseline. Compare scalar z1 versus joint ridge2D readouts, each at |log(k/k0)|<=0.7 and2.0. All24 arms use250 Adam steps, lr0.02, target predicted p=-0.15, loss relu(pred+0.15)^2 +0.001 mean(active log-edit squared)+0.01 relu(standardized latent displacement-3)^2; select minimum ML loss including step0. Fixed nodes/topology, positive springs positive and zeros unchanged. Finite-difference initial gradient checks precede optimization. This is a response-calibrated latent proxy experiment, not decoded-position autonomous engineering or multi-model-seed evidence.

Controls: twelve matched-magnitude permutation/random-sign edits, seed123+index, one per network/bound using the ridge edit magnitude; six unchanged originals. All42 final graphs and edits are hashed in a manifest after every optimizer finishes. Optimization process prohibits child processes. A separate afterok job checks every hash before any LAMMPS evaluation, then verifies every graph at120 quasistatic compression increments using the existing physical first/last unwrapped boundary-node estimator. No final physical score affects an edit, early stop, restart, or winner. All arms, failures, source-wise counts and targets are retained. Failed development evidence may motivate a separately documented new model/study; it cannot become simulator feedback within this frozen search or an untouched-test claim.

Runner `hpc/experiments/ml_only_engineering/run.py`; exact frozen recipes, job IDs, code hashes, optimization histories, gradients and final outcomes remain in the result directory. Pending outcomes; no target attainment claimed.

### Centered stiffness-gradient follow-up predeclared — 2026-09-08

Before inspecting any new final physical result, saved v1 ML edits revealed almost uniform log-stiffness increases: ridge Reid2 mean0.1848/SD0.0002, low-T70 mean0.1311/SD0.0001. Existing uniform-scale physical controls in the historical endpoint study already establish negligible response change under that operation. This is a concrete proxy-exploitation diagnosis, not evidence of negative engineering success. The two v1 bounds did not bind most solutions and often produce identical designs; do not interpret repeated bound outcomes as independent evidence.

New independent study `notebooks/results/ml_only_engineering_v2/` preserves all v1 artifacts. Same pristine six development networks, frozen model/readouts,24arms and250-step ML-only recipe, but subtract active-gradient mean and project every Adam update onto zero-sum active log stiffness intersect the predeclared ±bound box. Bisection50 iterations, mean tolerance1e-6; this preserves geometric-mean stiffness and prevents uniform scaling. Random controls now permute signed centered edits, preserving mean, norm and bound. No current LAMMPS score chooses this edit or any iterate. All42 finals freeze before separate final verification. One AE seed; this remains development, not a held-out reliability demonstration. Exact jobs/recipes/hashes in each result directory.

V1 optimizer4675254 completed; initial verifier4675255 failed before any physics due an old frozen helper signature lacking compression_steps. Verification-only retry4675268 uses the already checked current endpoint helper and unchanged final-manifest SHA683ae6...f879b; complete hashes retained in artifacts. No optimizer rerun to repair this failure.

### Fixed-budget latent-direction engineering predeclared — 2026-09-08

Inspection of frozen v2 optimizer histories, before inspecting v2 physical results, showed that reaching the surrogate target often terminated useful edit growth through ML checkpoint selection: Reid53 z1 reached predicted-0.149 at step11 with max|log edit|0.232 despite bound2. This does not imply actual physical target attainment. Predeclare independent v3 to test the direction itself without trusting the affine readout's response magnitude.

`notebooks/results/ml_only_engineering_v3/`: same six pristine development networks × z1/ridge2D × fixed bound2 =12ML finals, six signed-permutation controls and six originals (24total). Use150 steps of centered max-absolute-normalized gradient descent, step0.05; project active log edits to zero mean and ±2 after each step. Minimize the signed frozen affine readout directly; select minimum ML objective, with no predicted-p target stopping, response-based acceptance, or latent-distance penalty. The extrapolated readout is a directional proxy and cannot be assumed to be a calibrated p-ratio prediction. All finals freeze and hash before any physical check, same120-step endpoint protocol. This tests the user's AE-only gradient-guidance hypothesis with stronger network redistribution; it is not a new AE fit, a simulator-guided continuation, or an untouched test.

Runner `hpc/experiments/ml_only_engineering/run_direction.py`; all prior protocols/results retained. V2 jobs4675275/4675276; v3 exact job IDs and frozen hashes in submission_status.json. Root owns scientific logs, operational agent owns queue/status artifacts.

### Development results collected; new-network and position-rollout checks — 2026-09-08

All three initial studies completed and every graph/edit/recipe hash passed the collector: v1 42/42, v2 42/42, v3 24/24 valid final physical evaluations (108total, including controls/originals). All60 initial finite-difference gradient checks passed; largest relative error0.0001273194. Inline notebook `notebooks/network_design/02_ml_only_engineering.ipynb` executed and reread with two saved PNGs. Full review, all final rows, source summaries and provenance: `notebooks/results/ml_only_engineering_review/`.

Uniform-scaling v1 mostly leaves physical p unchanged despite negative readout values. Centered target-based v2 improves all3/3 Reid and3/3 low-T for each policy/bound, beating each matched random control. At bound2 z1 meanchange=-0.0737017 Reid/-0.0313755 low-T. Stronger fixed-budget v3 improves3/3 Reid, with2/3 negative and1/3 <=-0.1 for each policy, but improves only1/3 low-T and worsens70/78. v3 z1 physical original→final: Reid2 .2243479→.0807838,53 .0904982→-.1446267,59 .1636564→-.0139853; low-T70 .3312937→.3500932,78 .3568775→.4193429,96 .1891007→.1293262. All measured after designs froze; no optimization feedback. One frozen AE seed; cannot guarantee reliability from these development networks.

V4 transfers the exact predeclared v3 method to previously unengineered AE-validation IDs: Reid57/15 and low-T105/152, selected in original validation order outside both readout calibration and previous six design IDs. No response-based ID selection. Prepare4675313 passed exact zero-error parity on normalized/physical-reference fields against known initial graphs, using only raw frame0, unchanged source readouts and frozen protocol compression scale. Optimize4675314 and final-only verify4675315;8ML+4signed-permutation+4original=16finals. This is new engineering-validation evidence, not the project's reserved final test. Artifacts `ml_only_engineering_v4/`.

Low-T follow-up `ml_only_engineering_rollout_v1/` replaces the latent-response proxy with differentiable physical positions from frozen AE+propagator. Two existing reference-simplification R16/D4/context16 bundles (seeds786/123), each low-T70/78/96. Model-only restore uses stored weights/stats and loads no trajectories; only pristine initial references are supplied. No fitting or response checkpoint selection (propagator metricnull). Single initial encoding and static16D context,199 autonomous delta-MLP steps, tensor decoder and exact physical inversion. Fixed original quantile0.10 sidegroups; endpoint estimator min reciprocal ratio with epsilon1e-12. The old published rollout R² was normalized-coordinate and is not reused as primary physical performance evidence.

Six ML finals plus six signed-permutation controls and three originals (15). Optimize150 Adamsteps lr0.02, zero-mean log-stiffness box±2, minimum downstream ML loss: relu(predicted physical p+0.15)^2 +.001mean(active logedit²) + squared relative predicted-compression error against the fixed protocol. Future observed positions and response labels are never used. Centered finite-difference gradient check before each arm; all model parameters frozen. Every final graph/edit hashes before separate120-step LAMMPS verification. Root review fixed physical edge-order mapping and exact metric/normalization contracts; both model-only loader checks passed. This is a new development hypothesis, not simulator-guided correction of a frozen design. Exact PBS identities, hashes, failures and results retained with artifacts.

### ML-only engineering collection complete — 2026-09-08

All authorized chains finished. Final verifiers4675268(v1),4675276(v2),4675287(v3),4675315(v4),4675329(position-rollout) exited0. Final counts42/42+42/42+24/24+16/16+15/15=139/139 valid, including originals/controls. Optimizers4675254/4675275/4675286/4675314/4675328; new-reference preparation4675313. Original failed verifier4675255 retained. Complete code/recipe/final-graph hashes, job resources and statuses are in per-study ledgers and `ml_only_engineering_review/provenance.json`. All five final manifests were checked against every graph and edit before collecting; no simulator feedback entered any optimizer.

Fresh engineering-validation v4 (same fixed-budget v3 method without tuning): both policies improve2/2 Reid and2/2 low-T, each beating its matched random control. z1 originals→finals: Reid57 +.0217835→-.1018818 (new conversion); Reid15 -.1444513→-.2735943; low-T105 -.6033957→-.6224349; low-T152 -.5039466→-.5534723. The latter three were already negative and must not be counted as positive-to-negative successes. Across development+new-validation Reid, z1 converts53/59/57; one frozen AE seed, not a universal reliability guarantee or reserved final-test claim.

Position-based autonomous engineering did not repair positive-p low-T intervention prediction: each seed improves only1/3 (network78), worsens70/96, and yields0/3 negative designs. Final mean physical changes: seed123 +.006838878; seed786 +.007835880. All6/6 ML final checks and6/6 initial autograd finite-difference checks valid (max relative gradient error .0110511). Post-fit physical-position rollout p-ratio R² at199 on each model's own three final designs: seed123 -9.284894, seed786 -2.369055; seed mean±SD -5.826975±4.890237, valid6/6 total, each3/3. MAE .203036/.107761 (mean±SD .155398±.067370). This is an edited-design development diagnostic after ML optimization, not the standard shared validation cohort or evidence that AE reconstruction fails. Estimator, original physical side groups, inversion and future-state exclusion are retained in the frozen runner.

Final evidence: all_final_results.csv, source_summary.csv (separate conversion/new-target counts), rollout_engineered_prediction_metrics.csv, and review.md under `notebooks/results/ml_only_engineering_review/`. The practical demonstrated result is AE-gradient-guided engineering without simulator feedback, with useful new-network transfer; reliable negative-target attainment on positive-p low-T networks remains open. No additional jobs are left running by this task.

### High-initial-response engineering with position edits — 2026-09-08

User requested harder high-p original networks and explicitly requested position-only edits, noting dePablo may be more naturally engineered through geometry. Clarified the gradient path: differentiate the frozen latent readout through z=encoder(original-network parameters), then update physical spring stiffnesses and/or initial positions; AE weights remain fixed. No direct optimization of a detached latent code and no LAMMPS edit selection.

Read-only screening of the original engineering AE declared train+validation pool:40/40 Reid and40/40 low-T valid existing raw frame0→199 physical endpoint-side estimates (quantile0.10, epsilon1e-12). No new physics or final-test access. Highest two/source: Reid65(train) .387292660 and93(validation/readout-calibration) .381288253; low-T50(train) .368578922 and13(train) .364224068. These are response-selected targeted development examples, not held-out generalization. Existing raw trajectories differ slightly from final quasistatic replay; screening values are not substituted for final physical baselines (known Reid53 difference .004553). Exact80 rows, split IDs, raw dataset hashes, valid counts, and parity comparisons: `high_pratio_engineering/screening/`. Screen4675379 completed rows but failed JSON serialization; repair4675384 preserved the failure and completed records. No LAMMPS screening.

Study `high_pratio_engineering/study_v1/`: four original networks × stiffness bound2 / stiffness bound4 / positions-only (stiffness fixed) / joint bound4 =16 ML designs; eight signed-permutation stiffness controls and four originals,28finals. Same frozen2D AE seed786 and z1 readout. Up to150 centered/block-max-normalized gradient steps with ML-only backtracking at .05/.025/.0125/.00625; choose minimum visited ML score. Stiffness log edits have zero active mean and prescribed bounds. Geometry edits scale q by0.25 original shortest incident bond and bound node displacement by0.5 that length. Fixed original box/topology and quantile0.10 boundary nodes; interior coordinates clipped inside boundary-side extents. Reject only nonfinite/invalid geometry, bond lengths outside0.25..1.75 original, pair separation below0.1 original minimum, or changed measurement groups. Recompute normalized and physical edge vectors/lengths, physical reference positions and spring rest lengths consistently. Position-only export asserts exact original stiffness; spring-only export asserts exact original positions. No matched random-geometry control is claimed.

Prepare4675403 passed exact zero-error initial preprocessing parity. Root review corrected source-wise bundle counts, global freeze gate, positional projection units, zero-edit encoder parity, early per-block finite-difference gates, graph export checks and verification provenance. Optimize4675413 exited0 (1:02,731556kB), froze all28 designs before verify4675414 started. The verifier checks every graph/edit hash, then reports all120-step physical replays; no physical score can modify any design. Exact immutable code, recipes, histories, geometry/gradient checks, jobs and final results retained in study artifacts. Pending physical outcomes; no high-p target success claimed yet.

### High-p geometry results and feasible-gradient follow-up — 2026-09-08

Study_v1 verifier4675414 exited0,28/28 valid (7:18,745396kB), frozen global manifest SHA e59399182e9c719fa484dd112efd3f5da11bc0b930640ebfe5dc75b8c7f0e373. Positions-only improves all four high-p originals; neither stiffness-only bound improves any of them. No negative final among the16ML arms. Exact source-wise counts and all final values are retained in study_v1/results.csv and review artifacts. Position-only original→final:
- reid 65: 0.387370291 → 0.335775118.
- reid 93: 0.381988592 → 0.294475216.
- depablo_low_temp 13: 0.363512008 → 0.317482605.
- depablo_low_temp 50: 0.369090053 → 0.310853048.

Terminal ML-only diagnosis in geometry_stop_audit.json: all four positions-only arms stop because1–2 overlong bonds involve2–4 nodes; none of the tested backtracks violates pair separation or measurement-side membership. Their displacement radius is also active. This establishes constraint stopping, not an ML optimum or a failure of geometry gradients.

Predeclared study_v2 reuses the same pristine high-p development originals and frozen2D/seed786 AE plus historical z1 response-calibrated directional proxy. No training, final-test access, new screening, or simulator-selected starts. Compare positions-only movement radius0.5 versus1.0 original shortest incident bond, and joint radius1.0 with small zero-mean log-stiffness bound0.25. Keep box/topology/side-boundary nodes fixed, original bond-length interval0.25..1.75 and minimum pair separation0.1 original minimum.400 ML gradient steps, block-max normalization and backtracking .05/.025/.0125/.00625. When geometric rejection blocks every backtrack, temporarily mask the offending bond/pair endpoint position gradients and retry, up to16 mask retries per iteration; reset mask next iteration. Choose only the minimum frozen readout over valid iterates. Recompute initial geometry and rest lengths consistently; position-only stiffness remains exact.

Four networks×3MLarms=12ML; four original baselines and four permuted movable-node q-field controls from the radius1.0 position arm give20finals. Control seeds123+networkID; project to position constraints and geometrically halve displacement until valid (maximum25 attempts), without ML or physical score selection. Report resulting norm ratio; these controls are not necessarily norm matched. All20 design/edit hashes freeze globally before a separate120-step LAMMPS verifier. Initial finite-difference and zero-edit parity gates retained. Runner run_geometry_feasible.py; immutable code, recipes, hashes, jobs and failures in high_pratio_engineering/study_v2. Reused development graphs and completed prior-study diagnosis are explicit; this is not a new blind test. Pending outcomes.

### Position-gradient budget extension predeclared — 2026-09-08

Study_v2 optimizer4675464 completed all12MLarms at400 steps, exit0,3:41,728564kB; verifier4675465 is running. All eight position-only ML histories still improve over their final50steps (signed-readout decrease0.0111..0.0988). First Reid physical rows have already been observed:65 position radii0.5/1.0 give.07017083/.05737785 from.38737029;93 gives.06916634/.14486576 from.38198859. No pretense that this new declaration precedes all prior physical feedback: this is a separate targeted DEVELOPMENT study, with no reused graph becoming a fresh blind test.

Study_v3 tests whether additional ML budget improves the geometry direction: same four pristine originals, frozen2D/seed786 model/readout, both position-only radii0.5/1.0, exact stiffness preservation, same feasibility-mask/backtracking/geometry constraints, and1200 steps instead of400. Both radii retained for every source; no physical winner selection or edited starting graph. EightMLfinals + four original baselines + four feasible permuted radius1.0 geometry controls =16. Control seeds123+ID and deterministic geometry-only shrinking unchanged. Freeze/hash all16 before separate120-step final LAMMPS verification; no physics within optimization, stopping, or iterate selection. Initial parity/finite-difference checks retained. Runner run_geometry_long.py; all artifacts/jobs/failures under high_pratio_engineering/study_v3. V2 remains unchanged and all its outcomes will be reported.

### Feasible-position gradient results collected — 2026-09-08

Study_v2 completed: optimizer4675464 exit0,3:41,728564kB; final verifier4675465 exit0,4:02,743868kB.20/20 valid, frozen global manifest SHA d4ddd07c734f522bf98ffa1ff3f6d871493b4c2da996ef118dfd52470c267d59. All12ML designs improve physical p-ratio, including2/2 per source in each arm;0negative/12ML,0<=-0.1/12ML. All arms completed400 steps, versus early stops in study_v1. One frozen AE seed786; these are targeted reused development networks, not seedreplication or held-out generalization.

| Source | ID | Original | Positions radius0.5 | Positions radius1.0 | Joint radius1.0/logbound0.25 |
| --- | --- | --- | --- | --- | --- |
| reid | 65 | 0.387370291 | 0.070170830 | 0.057377853 | 0.098892393 |
| reid | 93 | 0.381988592 | 0.069166343 | 0.144865762 | 0.162630061 |
| depablo_low_temp | 13 | 0.363512008 | 0.152710004 | 0.089129130 | 0.083955714 |
| depablo_low_temp | 50 | 0.369090053 | 0.123320862 | 0.102165792 | 0.100380069 |

Radius1.0 position edits beat all4 feasible permutation controls, but controls required deterministic0.5 shrinking and have q norm ratios0.491756/0.492132/0.497807/0.499383; do not claim matched edit magnitude. No physical result selected an iterate, restart, or final winner. Position-only stiffness is exact and positive springs/topology/box/measurement groups preserved; physical rest lengths follow edited geometry. R² from reconstructed/predicted positions is not measured by this AE latent-proxy engineering study; do not represent the historical z1 affine score as a calibrated position rollout prediction.

All results/controls, immutable hashes, finite-difference/identity checks, optimizer histories and physical inputs/dumps are retained under high_pratio_engineering/study_v2. Notebook03 is being updated with every declared arm. The separately predeclared1200-step study_v3 is running, with no target attainment assumed.

### Longer position-only budget results collected — 2026-09-08

Study_v3 finished: optimizer4675486 exit0,7:54,731244kB; verifier4675487 exit0,3:20,743640kB.16/16 valid, frozen global manifest SHA50ba6dab486089f0a7542a24f3520c75daa933a0e2a2c3b6ce2590ac1b39ed4b. All8ML finals improve physical p-ratio (2/2 per source in each radius), but0/8 are negative and0/8 reach-0.1. Every corresponding1200-budget final also improves over its400-budget physical result. Seven arms reached1200; low-T13 radius1.0 stopped at827 after no accepted feasible descent. One frozen AE seed786; no independent seedreplication claim.

| Source | ID | Original | Positions radius0.5 | Positions radius1.0 |
| --- | --- | --- | --- | --- |
| reid | 65 | 0.387370291 | 0.049448836 | 0.009107695 |
| reid | 93 | 0.381988592 | 0.043504126 | 0.059287910 |
| depablo_low_temp | 13 | 0.363512008 | 0.138949904 | 0.077234588 |
| depablo_low_temp | 50 | 0.369090053 | 0.102983994 | 0.061259904 |

Both source/radius groups have valid2/2,total2; source means and exact control normratios in source_summary.csv and review.md. Radius1.0 ML designs beat all4 feasible permutation controls, which remain non-norm-matched. None of these targeted high-p cases is a negative-p success: Reid65 final+.009107695 is still positive. Gradients guide physical initial-position edits through the frozen AE latent readout; stiffness is held exactly fixed, topology and box fixed, rest lengths recomputed. This substantially improves response without simulator feedback, while reliable high-positive→negative target attainment remains open. Increasing ML budget helps these cases but does not establish calibrated response prediction or generalization to unobserved interventions.

All three high-p studies preserved:28/28+20/20+16/16=64/64 valid physical evaluations including repeated originals and controls, not64independent networks. All model training remains positions/trajectories/graph only; these engineering studies use an explicitly historical response-calibrated downstream z1 proxy, not physical-position rollout R². Prior five ML-only studies remain139/139; no result was overwritten. Notebook03 includes each study, all declared arms/controls, source counts and saved inline figures. Both v2/v3 PBS chains complete; no further engineering jobs left running by this task.

Notebook03 final execution reread confirms4 embedded PNGs and0 execution errors. Final project check passed with donor venv:11notebooks,83codecells,8datafiles. An initial check with defaultpython failed because graph_utils is absent from that interpreter; reran the established project environment, without changing code or results.

### Joint-parameter follow-up predeclared — 2026-09-08

User requested joint initial-position and stiffness engineering, plus thermal mixed-T engineering with a linear fit across frames. Existing joint400-step bound0.25 evidence does not establish superiority over positions alone: it improves all4high-p cases but is worse than radius1.0 positions-only in3/4. Predeclare a longer joint-budget comparison to the completed positions1200 study; reuse its four pristine development originals, frozen2D AE seed786 and historical downstream z1 readout, fixed box/topology/side nodes and radius1.0 geometry limits. Joint log-stiffness bounds0.25 and1.0, zero active mean, same1200-step normalized gradient/feasibility-mask/backtracking algorithm and minimum valid ML score selection. EightML + four originals + four controls =16finals. Controls permute both centered log stiffness and movable-node q fields from joint bound1, seed123+ID; geometric-only q shrinking allowed and reported, stiffness permutation preserves norm/mean. No physical score selects a start/iterate/arm, all final hashes freeze globally before final120-step quasistatic verification. This is reused targeted development, not held-out reliability or independent model-seed replication. Artifacts high_pratio_engineering/joint_v1, runner run_joint_long.py. Mixed-T protocol audit is separate; endpoint quasistatic validation will not be mislabeled thermal verification.

### Mixed-temperature joint engineering predeclared — 2026-09-08

User requests thermal/mixed-T original-network engineering and a linear fit across frames for p-ratio. Protocol audit found no original mixed-T thermostat/integrator/damping/timestep/mass/deformation provenance in repository/donor/MetaForge; the user has been asked for its location. Existing endpoint helper is athermal and will not be used or labeled as thermal verification. Optimization and existing-trajectory metric evaluation can proceed independently; no thermal physics may run until its protocol is explicitly resolved and frozen.

New mixed-T validation cohort: first ID per temperature in the existing20-ID validation ordering, without response screening: T0.1→47,1→68,5→140,10→161,20→242,30→268. Use only these original frame0 graphs for optimization, no final-test access. Transfer the unchanged historical low-T z1 affine readout and frozen2D AE seed786; no mixed-T response calibration or model fitting. This is a zero-shot directional proxy, not autonomous position prediction. Source was excluded from original AE/readout fitting; six networks, one perT, do not establish reliability across networks at eachT.

Compare position-only versus joint log-stiffness bound0.25 for all six originals, fixed geometry radius1.0 local shortest bond, original box/topology/side nodes, bond lengths0.25..1.75 original, minimum separation0.1 original minimum, zero active-mean stiffness logs. Same1200-step normalized-gradient/ML-backtracking/feasibility-masking protocol, minimum valid historical ML score, finite-difference and zero-edit parity gates. TwelveML + six pristine originals + six controls =24 globally frozen designs. Controls independently permute joint-bound0.25 centered logstiffness and movable-node q, seed123+ID; geometrically halve q as needed, report resulting normratios. Allgraph/edit hashes freeze before any thermal verification. Runner run_thermal.py and prepare_thermal.py, artifacts thermal_joint_engineering/study_v1. No optimization depends on future observed frames or physical response.

Thermal primary estimator predeclared: fixed initial quantile0.10 boundary groups, physical unwrapped node coordinates; width/height engineering strains. OLS fits WITH intercept separately versus saved frame index over ALL200 frames: eps_x=a_x+b_x*t and eps_y=a_y+b_y*t; p=-b_y/b_x for known x-driven compression. No reciprocal-min branch, response-selected window, endpoint substitution, or strain gate. Require>=8 finite pairedframes and fitted axial total strain magnitude>=1e-3; report all valid/total and invalid cases. Existing raw trajectories unwrap fractional-coordinate minimum-image increments through each evolving box, assumption below halfbox/frame explicitly retained. Fit R² for strain-versus-frame is a temporal noise diagnostic, not model p-ratio prediction R². Final thermal verification should use three declared noise seeds with matched original/edited protocols; model seed remains786 and thermal-seed SD must not be mislabeled model-seed SD. Exact thermal simulation protocol remains pending provenance, no physics submitted for this cohort yet.

### Thermal metric and diagnostic precision repair — 2026-09-08

Implemented thermal_metric.py with NumPy and differentiable Torch versions of the predeclared two-OLS slope-ratio estimator. Affine negative-p synthetic example, noisy NumPy/Torch parity (absolute difference5.55e-17), and degenerate invalid-return tests pass. Existing selected validation trajectories:6/6 valid, each200/200 finite, graph time0..299893 at uniform1507 cadence, x-box ratio0.9700107. Measured original p at T0.1/1/5/10/20/30 is +.210245/-.480160/+.352443/+.327606/-.245716/-.626450. Three are alreadynegative; do not count them as engineering conversions. Artifacts thermal_engineering_audit/raw_linear_fit.csv, metric_checks.json and raw_linear_fit_recipe.json preserve exactsourcehashes/IDs/counts/estimator and temporal-fit R² (not modelpredictionR²).

Thermal preparation4675596 completed exit0,31s,1128664kB with frame0 physical inversion checks. Initial optimizer4675597 failed at first initialposition FD gate before ML search/globalmanifest/physics (original reference export exists). Failure/recipe/snapshot retained at thermal_joint_engineering/study_v1. Diagnostic finite_difference_precision_audit.json reproduces weak derivative7.25319e-5: float32 central-difference relative error grows from.000178 at eps.03 to.02649 at.01,.04127 at.003,.18874 at.001,.26987 at.0003. This is numerical cancellation, not evidence of a wrong gradient.

Repair uses fixed central-difference probe epsilon.01 for all initialchecks, retaining tolerance.1 and exactsame model, gradient-based optimization, seed, objective, budget and cohort. All12 source/block preflightchecks pass, maximumrelative.02649375; fulltable finite_difference_repaired_preflight.json. Retry artifact thermal_joint_engineering/study_v1_retry preservesfailedstudy_v1 and reusesits preparedbundle readonly; no data/model fitting rerun. All24 designs stillfreeze beforeany thermalphysics. Thermal verification protocol remains unresolved; no thermal verifier submitted. Exactretryjob/snapshot hashes inledger.

### New finite-temperature verification protocol frozen — 2026-09-08

User explicitly said Continue after the missing original generation recipe was reported. Proceed with a separately declared NEW thermal test, not a claim of dataset-protocol reproduction. All24 mixed-T originals/ML edits/controls already froze globally before this declaration, and no thermal physical outcomes have been inspected. Exact protocol and dependency hashes: thermal_joint_engineering/study_v1_retry/thermal_protocol.json.

Six recordedT values0.1/1/5/10/20/30K; three initial Maxwell velocity seeds786/123/456 paired across originals and variants. Metal units,2D enforce2d, periodic box, harmonic springs and zero-angle coefficients, explicit mass1e6amu overriding the athermal init.mod dummy mass1e-20. Timestep.001ps. Initial minimization with fixedx/y-zero-stress relaxation, then100000steps thermal equilibration with Nose-Hoover temperature chain3/damping1ps and y-only zero-pressure barostat chain3/damping10ps. Reset timestep; apply x scale.9700107 continuously over299893steps with remapx, y still zero-pressure coupled. Save200 frames at0,1507,...,299893; these cadence/box-scale choices match observedmetadata but thermostat/mass/timestep provenance is NEW. All-frame OLS estimator stays predeclared; fixed initial frozen-graph sidegroups andphysicalunwrapped xu/yu positions. Eachsimulationtimeout600s; allfailurespreserved. ReferenceLAMMPS fix_nh/fix_deform/fix_enforce2d officialdocs linked inprotocol.

24designs×3velocityseeds=72finalevaluations (36ML,18control,18original). Six temperaturejobs, each1CPU2GB free:shared, nohostpins,3hwalllimit; independentoutputs and noMLcallbacks. Hash everydesign/edit, modelrecipe, metric, verifier, LAMMPS and exporter beforephysics. Thermal-seed mean/SD and pairedchanges use newprotocol originals, not previously observedrawtrajectory values. This test measures newthermal response; it cannot establish exact replay of originaldataset conditions or broadreliability from one network perT. No physical score can revise any frozen design, objective, start, or final-selection rule.

Metric validity flag repaired to require finite p and axial-fit threshold, rather than merely eightfiniteframes. The earlier synthetic affine example was positive+.3 (previous prose callingitnegative was inaccurate); retainedthatcaseandaddedanexplicitnegative-.3 case. No originalbaselinevalueoroptimization changed.

### Longer joint results collected — 2026-09-08

Joint optimizer4675575 completed exit0 (7:34,730848kB), verifier4675579 completed16/16 valid. Frozen manifest SHA b5093507438100be4019ea6fe02a41023aa9cab2618292fed37f6339a5a15fb3. All8ML finals improve versus matched pristine originals (2/2 per source/bound);0negative,0<=-0.1. Samefourtargeteddevelopmentnetworks/frozen2D AE seed786, no held-out reliability or independent-seed replication. Controls and all physical rows retained.

| Source | ID | Joint bound0.25 | Joint bound1 | Prior matched positions-only |
| --- | --- | --- | --- | --- |
| reid | 65 | 0.060263458 | 0.129303203 | 0.009107695 |
| reid | 93 | 0.079099037 | 0.095987647 | 0.059287910 |
| depablo_low_temp | 13 | 0.073764503 | 0.093734492 | 0.077234588 |
| depablo_low_temp | 50 | 0.061050176 | 0.041388250 | 0.061259904 |

Joint optimization is not uniformly superior: bound0.25 is slightlybetter on bothlow-T examples andworseonbothReid; bound1 isbetteronlow-T50 andworseontheotherthree. This comparison reportsbothpredeclaredbounds without selecting a physical winner. Modelscoreimprovement is not calibrated intervention-response prediction. Exact source summaries in source_summary.csv; alljob/code/recipe/designhashes and failures in joint_v1. Completedpos/geometryevidence retained separately.

### Thermal training/calibration overlap audit — 2026-09-08

Exact frozen AE provenance recovered: reid_lowT_2d_s786, checkpoint SHA62d9476c47f4024df0395d7dc14d4942a86f5c1baafb438ceb3ec6681be6dcc7. Audited actual20Reid+20low-T AE training initialgraphs and20historicalreadoutcalibration initialgraphs, all40/40 and20/20 valid. Six selected mixed-T graphs have0/6 matches against AE training under node-count/canonical undirected node-labelededgepairs and pairedstiffness hashes. Two match historicallow-T readout-calibration graphs exactly in topology/stiffness: mixedT47→lowT47 andmixedT161→lowT161. Their geometry hashes differ. The otherfourmixedT IDs68/140/242/268 have no calibration match. Exact IDs/hashes/provenance and limitations at thermal_engineering_audit/training_overlap.json.

Therefore source exclusion from AE/readout fitting does NOT establish unseen-topology calibration disjointness. In particular the0.1K negative-conversion case is a new thermal-state/intervention result on a calibration topology, not an unseen-topology reliability demonstration. Allsixpredeclaredcases will stillbe reported; no replacing a case or physicalwinner selection. Hash mismatch is not proof of arbitrary graph-isomorphism disjointness.

### Ten positive Reid originals: comparison predeclared — 2026-09-08

User requests optimizing10positive-p Reid networks. Existing screening has EXACTLY10positive Reid validation originals, so includeall10inpreviousvalidationorder:93,84,96,97,52,59,57,53,2,74. No newresponsewindow/threshold tuning or reservedfinal-test access. Original existingtrajectoryp ranges+.0229..+.3813; rawscreen isselectionevidence, notsubstitutionforthe matchedfinalphysicalbaseline. cohort.json under reid_positive10/study_v1 preserves IDs, rawscreenvalues, split, screenhashes andper-IDreadout/previousengineeringoverlap. FiveIDs93/84/96/97/52 were historicalreadoutcalibration; fiveIDs93/59/57/53/2 werepreviouslyengineered. This is a positive-response-selected validation/development cohort, not an entirelyunseen-network reliabilitybenchmark.

Freeze existing2D AE seed786 andhistoricalReid z1 affine readout, nofit/newexpert supervision. Comparepositions-only versusjointlog-stiffnessbound.25 foreachnetwork, physicalmovementradius1.0 shortestoriginalincidentbond(qradius4), sameexistinggeometry/box/topology/fixedboundary/zero-meank constraints.1200steps normalizedgradientdescent withMLbacktracking .05/.025/.0125/.00625 andup to16offendingnode-mask retries;minimumvalidMLscore selection. InitialFDprobeeps.01 andtolerance.1 pluszeroeditencoderparity; numericalFDchange followsdocumentedfloat32precisionaudit, notoptimizerchange. Ten joint controls independentlypermute centeredstiffnesslogs andmovableq fields, local seed123+ID, deterministicgeometry-onlyq halvinguntilfeasible, reportnormratios.

10originals+20ML+10controls=40globallyfrozendesigns. Reuse the identical completedReid93 positions1200/radius1.0 andjoint1200/bound.25 outputs, with exactreference/design/hash checks andpriorhistoryprovenance; do not use either as an initialization for furtheroptimization. Reuse its original+twoMLphysicalmeasurements frommatchedprevious120-stepcompressionstudies;verifyitsnewjointcontrol andallother9networks onlyafterall40hashesfreeze (37newphysicalchecks+3reusedresults). All outcomesincludingfailuresreported; no physicalwinnerselection orsimulatorfeedback. Sameexistingquasistaticxcompressionwithyrelaxation, no thermalvelocityseeds. Modelp-ratioR² frompredictedpositions is notmeasuredbythishistorical-latent-proxy engineering comparison;actualphysicalnegative-p attainment is reportedseparately.

Implementation/submission delegatedTerra; routinewaiting/checks delegatedLuna perusercostpreference. Exactfrozenrecipes/code/model/datahashes/jobs/resources under reid_positive10/study_v1. No rootbusy-waitingforlongsimulations.

### Thermal final collection complete — 2026-09-08

All72/72 declaredfinalthermal checksvalid, each200/200frames.24globallyfrozenoriginal/ML/controlgraphs×threevelocityseeds, onefrozen2DAEseed786. Separate NEW thermalprotocol; notexactGTgeneratorreproduction. Independenttemperaturejobs4675668(T.1),4675672(T1),4675673(T5),4675669(T10),4675671(T20),4675670(T30). ProtocolSHA9ddf60e71b26f4de13bd2e3b1563ebfed40cb09e250c474da515f3cf8e29c708; finalmanifesthash and all source/result/dependencychecks retainedinverification_completed.json andperjobledgers. Luna completedroutinewaitingperuserrequest. Collector verifies all72keys/designhashes andperTcompletion/recipe/protocolhashes, retains temperaturediagnostics andpairedoriginal baselines. graph_utils/legacy side-group implementations agree24/24frozendesigns; sourcehashes in side_group_implementation_parity.json.

| T(K)/ID | Original mean p | Positions mean±SD | Joint mean±SD |
| --- | --- | --- | --- |
| 0.1/47 | +0.210048 | -0.077158±0.001640 | -0.080985±0.002242 |
| 1/68 | -0.491530 | -0.386559±0.004893 | -0.371974±0.006791 |
| 5/140 | +0.356487 | +0.063676±0.003651 | +0.056992±0.002873 |
| 10/161 | +0.329195 | +0.058582±0.007519 | +0.080616±0.005689 |
| 20/242 | -0.307993 | -0.312457±0.043528 | -0.313914±0.035210 |
| 30/268 | -0.568519 | -0.505509±0.061501 | -0.480874±0.057856 |

Everytableentryhas3/3validvelocityseeds. Bothmodesconvert thepositive0.1Koriginal to negativein3/3seeds (one topology);5K/10Kimprovebutremainpositive. Alreadynegative1K/30KcasesbecomeLESSnegativeunderbothMLmodes, showing unreliable directionthere;20Kmeanchangeissmallrelativevelocity-seedvariation. Bothmodesimprove meanresponseon4/6network-temperaturecases; no universalorheldout-topologyreliabilityclaim. 0.1K/10Ktopologiesoverlaphistoricallow-Treadoutcalibration (geometrydiffers); other4have noauditmatch, andall6have noAEtraininghashmatch. This is not graph-isomorphism proof. Three alreadynegative originals mustnotbecounted asconversions. Allcontrols/failuresretained; q controlsnotnormmatched. Thermal-seedSD isnotmodel-seedSD. Primarylinearfit usesphysicalunwrappedpositions andtwoOLSstrain-vs-frame slopes, notendpoint minreciprocal orlatentproxy magnitude. Thesephysicalengineeringresults arenotAEreconstruction/autonomousrolloutp-ratioR².

Final tables thermal_results.csv andthermal_summary.csv; notebook04 savedwithrawlinearfitplots andnewthermalphysicalmean±SDcomparison. OriginalmissingGTprotocol remains a limitation; userclarified desiretomatchGT and no exactGTreplay hasbeenclaimed. Newlyrequested10positiveReidbatch is a separate declaredquasistaticcompressioncomparison.


### Positive-Reid10 batch submitted (2026-09-08)

Prepared and submitted the predeclared ten-positive validation cohort: PBS `4675802.zeus-master` prepare (4 CPUs/2 GB/20 min), `4675803.zeus-master` optimize (4 CPUs/4 GB/60 min, afterok prepare), and `4675804.zeus-master` final verification (1 CPU/2 GB/60 min, afterok optimize), all `mendels_q`, `place=free:shared`. Frozen code/dependency manifest: `notebooks/results/reid_positive10/study_v1/code/dependency_manifest.json`, SHA256 `c1f0a39be84e6df1266b0d33d852d49af28f0007cf855052aa2124603cf0706a`. Submission ledger: `notebooks/results/reid_positive10/study_v1/submission_status.json`. Expected 40 final records: 10 originals, 20 ML designs, 10 permutation controls; 3 explicitly reused prior measurements and 37 new final simulations. No results available at submission. Luna assigned operational monitoring only; every predeclared result and failure will be collected.


### Adam comparison and numerical-check recovery predeclared — 2026-09-08

User requested Adam with patience-based early stopping. Separate `reid_positive10/adam_v1`: same ten positive validation/development Reid originals and frozen2D/seed786 AE plus historical response-calibrated z1 readout; no fitting or new response supervision. Both positions-only radius4 and joint radius4/log-stiffness bound0.25, starting pristine. Projected Adam lr0.02, betas(0.9,0.999), eps1e-8, max1200 updates; raw gradients masked for fixed/inactive variables, active stiffness gradient centered, no block-max normalization. Project each proposal onto existing bounds; halve the entire projected displacement up to16 times for geometric feasibility only. If none feasible retain current parameters; Adam moments still advance. Feasible updates may increase ML objective. Retain strict minimum feasible ML score including initial and final update. Early stop after50 consecutive updates without cumulative best improvement >1e-5 against the last patience-reset best; log steps, patience, acceptance scale, stopping reason. All20 ML finals,10 originals,10 signed-permutation controls globally hash before final120-step compression verification; no physical selection or feedback. Same constraints/control seeds123+ID and estimator as original comparison. One model seed, response-selected development cohort with documented calibration/prior-engineering overlap; not unseen reliability or position-prediction R². Compare achieved objective, physical p, executed steps and timing without presuming Adam is better.

Original baseline optimize4675803 stopped on Reid84 initial random-direction gradient check before completefreeze or any newphysics. Preserve study_v1 and traceback. Exact position derivative approximately−2.909e-5 versus float32 FD−2.533e-5 at epsilon.01, relative error.1292. Diagnostic sweep on all10 showed more severe cancellation on52 (random derivative3.816e-6, FD signflip at.01). Changed only numerical probe to the projected gradient direction within each editable block, fixed epsilon.01 and unchanged tolerance.1. All20/20 diagnostic probes pass, worstrelative error0.0005293. Exact scripts/results at reid_positive10/gradient_audit; no model/optimizer/physical feedback modification. Baseline recovery `study_v1_retry` reuses hashed prepared data readonly, preserves exact optimizer recipe and original93 completed-output reuse. Adam uses the same repaired gate. Neither starts from failedattempt edited outputs. All final outcomes including failures retained.


Baseline numerical-check recovery submitted: optimizer `4675877.zeus-master` (4CPUs/4GB/60min), final verifier `4675878.zeus-master` (1CPU/2GB/60min, afterok), mendels_q/free:shared. Dependency manifest SHA256 `5f8ac837119d7b79cca8faec4943271addfeb6e71f7c38d8661834374b04bae6` at `reid_positive10/study_v1_retry/code/dependency_manifest.json`. No optimization recipe change, no new physical result at submission.


Adam comparison submitted: optimize `4675889.zeus-master` (4CPUs/4GB/60min), verify `4675890.zeus-master` (1CPU/2GB/60min, afterok), `mendels_q`/`free:shared`. Immutable dependency manifest SHA256 `7545375110ec2d86dd5a8e812149be645d054220b7669ecdb55cde9d68fecf34` at `reid_positive10/adam_v1/code/dependency_manifest.json`. All40 Adam-study final measurements are new. Real frozen-model smoke on ID84 with2updates perarm passed graph constraints and4/4 exported hashes; both final-update minima retained. Compilation/diff checks pass; smoke harness/completion summaries retained in implementation_checks. This smoke checks implementation, not physical efficacy. Luna assigned both baseline-retry and Adam chains; no final results available at submission.


### Adam positive-Reid10 final verification collected — 2026-09-08

Optimizer4675889 and verifier4675890 completed;40/40 final physical evaluations valid (10originals,20ML,10controls),40/40 design/edit hashes verified. Frozen manifestSHA `1e048f27087d7302b631b2140e173db8374c7d6f24a0b1bb09de870c676751d6`; resultsCSV SHA `d47ae9d814a42f767b30dc23f5f2b607dec06818fac3f7e59951dfaa6482753d`. All20ML improve matched original p; positions4/10negative, joint5/10negative. Joint lower than positions10/10. No final physical feedback to optimizer or selection. OneAEseed786; no model-seed mean/SD estimate. Earlystopping100–128steps; all20arms patience50.

| Reid ID | Original | Positions | Joint |
| --- | --- | --- | --- |
| 93 | +0.381989 | +0.212933 | +0.208034 |
| 84 | +0.145364 | +0.034627 | -0.021575 |
| 96 | +0.071640 | -0.035423 | -0.083945 |
| 97 | +0.180748 | +0.079202 | +0.037481 |
| 52 | +0.278101 | +0.185327 | +0.154308 |
| 59 | +0.163656 | +0.041421 | +0.008403 |
| 57 | +0.021784 | -0.001141 | -0.023149 |
| 53 | +0.090498 | -0.045090 | -0.093093 |
| 2 | +0.224348 | +0.127923 | +0.091918 |
| 74 | +0.037355 | -0.005762 | -0.033035 |

All controls retained in results.csv. Highest-positive original93 remainspositive (+.382→+.208joint); improvedlatentproxy is not uniformly negativephysicalresponse. Cohort response-selected validation/development with previouslydocumented calibration/priorengineeringoverlap; not heldout reliability. This reports finalphysicalengineering endpointp, not autonomous-position or AEreconstructionp-ratioR². User now prioritizes Adam; no additional optimizer experiments launched.


### Overnight joint-Adam matrices predeclared — 2026-09-08

User explicitly requested joint-only hyperparameter sweeps on the same10positiveReid developmentnetworks and5mixedTnetworks pertemperature, then authorized expanded overnight experiments. No change to frozenAE2D/modelseed786 or historical source-specific z1 readouts; no expert supervision, checkpoint selection, or simulator-in-optimization. Everyarm starts pristine. Comparing physicalresults tomorrow is retrospective development recipe assessment, not selecting a simulator-verified winning design and calling it blindMLreliability. Preserve everyarm, control, failure andvalid/total.

Reid matrix: learningrate0.005/0.02/0.08 × zero-mean log-stiffnessbound0.25/1.0 × patience50/150 × feasibilityhandling whole_proposal/mask_blocking_nodes =24jointsettings × same10originals. Radius4 (maxone shortestoriginalincidentbond),1200Adamupdates,defaultbetas/eps, absolute min_delta1e-5, strictminimumfeasibleMLscore inclinitial/final. Mask-mode hypothesis: one blockedgeometry constraint may stop all Adam coordinates; after17wholeproposalhalvingattempts fail, freeze offendingnode positionincrements atcurrentvalues, retain stiffnessproposal, retryup to16maskpasses, resetmasknextupdate. Geometry/box/topology/fixedboundary/pair-distance/bondconstraints identical inbothmodes. NoMLdecrease-requiredacceptance, noLAMMPS duringoptimization. Matched controlsseed123+ID independentlypermute centeredlogstiffness andmovableq, geometricalshrinking only; notfullynormmatched. Reuse existingidenticalwhole/lr.02/bound.25/patience50 Adam10ML+10controls+10originalmeasurements onlyascompletedoutcomes, neverstarts.23newsettings×20finals=460newphysicschecks; globalfreeze validatesall490records (240ML,240controls,10originals) BEFORE ANYnewphysics. Shared4CPU2GB optimizerjobs,1CPU2GB final120stepathermalcompressionchecks. Compare recipe mean/medianphysicalp,negativecount,pairedchange,validcount andsteps; analyze matchedfactor effects ratherthan just minimumobservedwinner. One modelseed and response-selecteddevelopment cohort; noseedSD or unseenreliabilityclaim.

Thermal matrix fixed independently of Reid sweepoutcomes: A(lr.02,bound.25,patience50), B(lr.005,bound.25,patience150), C(lr.02,bound1,patience150), all jointwholeproposal Adam,radius4,max1200,min_delta1e-5. These compare existingrecipe,slower/longer, andwider/longer; jointfactorchanges inB/C are explicit, notisolatedcausalestimates.5networks eachT0.1/1/5/10/20/30 =30originals. Validation hasonly20rows(4/4/3/3/3/3), so retainall20 andfillto5each fromoriginaltrainorder(10additionaldevelopmentrows), noresponseaccess/selection or reservedtest. IDs: .1=[47,44,36,39,13];1=[68,96,70,78,77];5=[140,118,136,109,145];10=[161,191,158,198,151];20=[242,208,221,229,234];30=[268,250,262,265,290]. Cohortmanifest records exactsplits/hashes; training/calibration overlap audit separate. Directthermalframe0 edits through frozen low-TtrainedAE/readout, never correspondinglowTgeometry.30original+90ML+90controls=210globallyfrozendesigns; pairedvelocityseeds786/123/456 ->630thermalfinalchecks. Same previouslydeclaredNEWthermalprotocol: 100000equilsteps,299893xcompressionsteps,.9700107scale,200savedframes,metalunits/mass1e6/dt.001, NoseHooverT/ybarostat. OriginalGTgeneratorremainsunavailable; this is NOT exactGTreproduction. All-framephysicalp fromtwoOLSstrain-versus-frame fitswithintercept andfixedsidegroups; p=-slope_y/slope_x, finite/axialthreshold unchanged. 30pernetwork verificationjobs1CPU2GB4h; strictglobalfreeze beforeall. Report network-levelmeans/thermalSD with3/3seedvalidcounts, all5networks perT, alreadynegativeoriginals separatelyfrompositive→negativeconversions; modelseedSD unavailable. No finalphysicsfeedback, earlystop or restarts. Oneoptimizer4CPU2GB2h afterprepare4CPU2GB20min; no whole-node reservations/hostpins. Collectors andinline notebook plannedforovernightresults; routinewaitingdelegatedLuna.


Thermal30 cohort overlapaudit complete:30/30frame0graphs compared with40declaredAEtraininggraphs and20historicalreadoutcalibrationgraphs;4/30 exactnode-labelledtopology/stiffness matches AEtrain (mixed13/77/145/198),5/30matchreadoutcalibration (47/118/136/161/191). Geometry differs inallmatches. ExactIDs/hashes/method in thermal_adam30/study_v1/training_overlap.json; nonmatchdoesnotruleout relabelledisomorphism. This explicitlylimits unseen-networkclaims; no cohortreplacement or correspondinglowTgeometry used.


### Overnight joint-Adam pipelines submitted — 2026-09-08

Reid24: submissionledger `notebooks/results/reid_joint_sweep/study_v1/submission_status.json` SHA256 `30fef7fabac99532734650a8dc437174c002dbadb3fa70d7dcf84b1be825205e`; frozen dependency manifest `notebooks/results/reid_joint_sweep/study_v1/code/dependency_manifest.json` SHA256 `97bd8d405eb4b14bdc3a3695900337c8e5cf3a4bc31b4c1452a00ea4d5b47929`. Exact per-cell/per-ID jobidentities and dependencies retained inledger.

Thermal30×3: submissionledger `notebooks/results/thermal_adam30/study_v1/submission_status.json` SHA256 `238cb2d7920be739889bc2f74ba7f14946450a254fc3a1c8b5d9d530d1230ac2`; frozen dependency manifest `notebooks/results/thermal_adam30/study_v1/code/dependency_manifest.json` SHA256 `060656ca1911e4d5daa73d541365977423a5d71087ad063588f35052891f66cf`. Exact per-cell/per-ID jobidentities and dependencies retained inledger.

Reid:23newoptimizerjobs4676000–4676024(noncontiguous; exactmappinginledger), globalfreeze `4676025.zeus-master`,23verifiers4676026–4676048, collector `4676049.zeus-master`. 24totalsettingsincludingreusedbaseline;490finalrecords,460newphysics. All16initiallycompletedoptimizercells had no traceback at rootcheck; remainingrunning. Thermalprepare `4676061.zeus-master`, optimize `4676062.zeus-master`,30per-IDverifiers4676063–4676092, collector `4676093.zeus-master`;210designs/630checks. Allthermaljobsdependonglobalfreeze; nomeasurementsusedforoptimization. Sharedresources exactlyaspredeclared.

Implementation validation: Reidwhole+mask one-ID/two-update actualfrozenmodelsmokes pass; mixedT47 three-recipes/two-update smokeexports7/7designs withtemperature,positive/negative200frameOLS andthermalinputgeneration pass. Numericalall30gradientpreflight delegatedread-only. Rootreviewfixed per-kind control separation incollectors, missingpartialjobhandling, actualhash/protocolchecks. Notebook05 executes existingbaselineplot andpendingstates; separateclearlysynthetic/tmpfixture exercisesfull24×10heatmap andsixthermalpanels (3inlinePNGs) successfully. No syntheticvalueswritten toscientificresults.

Automaticreport job `4676102.zeus-master` runsafteranybothcollectors,1CPU2GB20min. It saves inline Editorial notebook `notebooks/network_design/05_joint_adam_sweeps.ipynb` and appends complete/partial source-wise results toindex/engineeringlog; reportcode andnotebook-at-submission hashes at overnight_engineering_report/code/hashes.json. No rootbusywaiting; Luna assignedoperationalmonitoring. No guarantee of completedresults ifjobsfail; failureswillremainvisible.


Thermal30 postlaunch numericalpreflight:60/60 projected-gradientchecks pass (30IDs×spring/position), maximumrelativeerror6.756186726659167e-05 atfixedeps.01, tolerance.1. Exact artifact thermal_adam30/study_v1/preflight_all_gradients.json. No optimization/design changes or simulationcalls.


### Overnight joint-Adam collection — 2026-09-08T21:55:49.842713+00:00

Reid: 490/490 valid finalrecords including explicitlyreused30baseline records; all24predeclaredrecipes reported. ResultsSHA `0b4a61d03fec6d36e96c4e7f11b54063ad9327b67317965692e4131cb1325f09`.

| Recipe | Valid/10 | Negative/10 | Mean physical p | Mean change | Mean steps |
| --- | --- | --- | --- | --- | --- |
| lr0.005_b0.25_p150_mask_blocking_nodes | 10/10 | 5/10 | 0.018804 | -0.140745 | 438.8 |
| lr0.005_b0.25_p150_whole_proposal | 10/10 | 5/10 | 0.023427 | -0.136121 | 415.8 |
| lr0.005_b0.25_p50_mask_blocking_nodes | 10/10 | 5/10 | 0.018994 | -0.140555 | 231.7 |
| lr0.005_b0.25_p50_whole_proposal | 10/10 | 5/10 | 0.023434 | -0.136114 | 218.0 |
| lr0.005_b1_p150_mask_blocking_nodes | 10/10 | 7/10 | -0.044039 | -0.203588 | 531.1 |
| lr0.005_b1_p150_whole_proposal | 10/10 | 7/10 | -0.041846 | -0.201394 | 408.9 |
| lr0.005_b1_p50_mask_blocking_nodes | 10/10 | 7/10 | -0.044030 | -0.203578 | 237.8 |
| lr0.005_b1_p50_whole_proposal | 10/10 | 7/10 | -0.041838 | -0.201386 | 224.1 |
| lr0.02_b0.25_p150_mask_blocking_nodes | 10/10 | 7/10 | 0.001969 | -0.157580 | 1059.7 |
| lr0.02_b0.25_p150_whole_proposal | 10/10 | 5/10 | 0.024531 | -0.135017 | 238.3 |
| lr0.02_b0.25_p50_mask_blocking_nodes | 10/10 | 7/10 | 0.003852 | -0.155696 | 255.1 |
| lr0.02_b0.25_p50_whole_proposal | 10/10 | 5/10 | 0.024535 | -0.135013 | 102.2 |
| lr0.02_b1_p150_mask_blocking_nodes | 10/10 | 7/10 | -0.062550 | -0.222098 | 1025.1 |
| lr0.02_b1_p150_whole_proposal | 10/10 | 7/10 | -0.045512 | -0.205060 | 214.6 |
| lr0.02_b1_p50_mask_blocking_nodes | 10/10 | 7/10 | -0.059057 | -0.218605 | 250.7 |
| lr0.02_b1_p50_whole_proposal | 10/10 | 7/10 | -0.045510 | -0.205059 | 102.3 |
| lr0.08_b0.25_p150_mask_blocking_nodes | 10/10 | 9/10 | -0.094140 | -0.253689 | 655.8 |
| lr0.08_b0.25_p150_whole_proposal | 10/10 | 5/10 | 0.024280 | -0.135269 | 168.4 |
| lr0.08_b0.25_p50_mask_blocking_nodes | 10/10 | 9/10 | -0.095372 | -0.254920 | 507.5 |
| lr0.08_b0.25_p50_whole_proposal | 10/10 | 5/10 | 0.024280 | -0.135269 | 68.4 |
| lr0.08_b1_p150_mask_blocking_nodes | 10/10 | 8/10 | -0.122526 | -0.282075 | 851.4 |
| lr0.08_b1_p150_whole_proposal | 10/10 | 7/10 | -0.048195 | -0.207744 | 169.9 |
| lr0.08_b1_p50_mask_blocking_nodes | 10/10 | 8/10 | -0.122549 | -0.282097 | 708.0 |
| lr0.08_b1_p50_whole_proposal | 10/10 | 7/10 | -0.048195 | -0.207744 | 69.9 |

Thermal: 630/630 valid finalchecks; observed 630, missing 0. ResultsSHA `d6859016c4bb676af167bc1114d35f8a891501b5a92aab58c914e8786d3ae61e`. Per-ID/per-recipe thermal-seedmeans/SD andvalid/total in thermal_summary.csv.

| T | Recipe | Networks with3/3valid | Mean of network p means | Negative network means | Positive→negative network means |
| --- | --- | --- | --- | --- | --- |
| 0.1 | A | 5/5 | -0.265950 | 4/5 | 2/3 |
| 0.1 | B | 5/5 | -0.262401 | 4/5 | 2/3 |
| 0.1 | C | 5/5 | -0.272733 | 4/5 | 2/3 |
| 1 | A | 5/5 | -0.084541 | 3/5 | 2/4 |
| 1 | B | 5/5 | -0.077722 | 3/5 | 2/4 |
| 1 | C | 5/5 | -0.079653 | 3/5 | 2/4 |
| 5 | A | 5/5 | -0.271839 | 3/5 | 0/2 |
| 5 | B | 5/5 | -0.267201 | 3/5 | 0/2 |
| 5 | C | 5/5 | -0.258963 | 3/5 | 0/2 |
| 10 | A | 5/5 | -0.168689 | 3/5 | 1/3 |
| 10 | B | 5/5 | -0.166363 | 3/5 | 1/3 |
| 10 | C | 5/5 | -0.146384 | 3/5 | 1/3 |
| 20 | A | 5/5 | -0.386252 | 5/5 | 1/1 |
| 20 | B | 5/5 | -0.384249 | 5/5 | 1/1 |
| 20 | C | 5/5 | -0.346581 | 5/5 | 1/1 |
| 30 | A | 5/5 | -0.401809 | 4/5 | 1/2 |
| 30 | B | 5/5 | -0.394691 | 4/5 | 1/2 |
| 30 | C | 5/5 | -0.340059 | 4/5 | 1/2 |

These are retrospective development comparisons on onefrozen2DAE/modelseed786 with historicalresponsecalibratedz1 readouts, not autonomouspositionp-ratioR² or heldoutreliability. Thermalspread isvelocity-seedSD, notmodelseedSD; identical separatelydeclarednewthermalprotocol, notGTgeneratorreplay. Thermalcohort20val+10train,4/30AEtrainingtopology/stiffnessmatches and5/30readoutcalibrationmatches; geometrydiffers. Allphysicalfailures/controls retained, no physicalfeedback duringoptimization orfinal-designselection. Notebook05 savedwithinlineEditorialplots.


### Morning interpretation and inline-plot repair — 2026-09-09

Both sweeps complete: Reid490/490validrecords (30reusedbaseline,460newchecks);thermal630/630validnewchecks. BestReidconversioncount amongpredeclaredrecipes: lr.08,bound.25,patience50,maskblockingnodes ->9/10negative,meanp−.0953715 (originalmean+.1595483), versusbaseline5/10,mean+.0245348. Bound1sameotherparameters haslowestmean−.122549 but8/10negative;thesearedifferentdevelopmentobjectives, notoneunambiguouswinner. ID93 remains+0.034060 in9/10recipe; ID52onlyslightlynegativeapproximately−.001, so conversioncountisnotallstronglyauxetic. Mainfactorpattern: blockingnodemask enableslargeLR; longerpatienceoftenaddsstepswithoutmaterialphysicalimprovement. Do notchooseindividualphysicallybestdesigns andclaimblindMLselection.

ThermalAandB eachlowernetworkmeanp30/30;C29/30. Eachconverts7/15initiallypositivenetworkmeans; alreadynegative15originalsarenotconversions. PerTconversioncounts .1:2/3,1:2/4,5:0/2,10:1/3,20:1/1,30:1/2. Allnetworkmeanestimates3/3validthermalvelocityseeds;onefrozenmodelseed. Node-labeledtopologyoverlap andnewthermalprotocol limitationsremainunchanged.

Overnight reportactuallysaved0inlinePNGs becausePBSforcedAgg. Correctednotebook05 with explicitinlinebackend andreran:3inlinePNGs nowpersist (completedbaseline,24×10Reidheatmap,sixthermalpanels), exactnewnotebookhash in overnight_engineering_report/inline_plot_repair.json. Originalreportcompletionpreserved;scientificCSVresults unchanged.


### Compact Adam presentation and frozen trajectory replay — 2026-09-09

User explicitly requests replacing engineeringnotebooks01–05 withonecompactAdam-only presentation: all10Reid and30mixedT before/afterstatistics, nooptimizerorpositions-onlycomparisons, severalactualtrajectoryGIFs. DisplaysameglobalReidrecipe lr.08/bound.25/patience50/mask for all10, thermalrecipeA for all30. Recipechoice isretrospectivedevelopment, notblindselection. Keepallcohortoutcomesincludingnonconversions andallhistoricalresult/log/provenanceartifacts.

FiveReidGIFs arefirstfivecohortIDs93/84/96/97/52, includeshardpositive93 andbarelynegative52. Endpointevaluationssavedonlyinitial/finaldumps; rerunexactfrozenoriginal+finalgraphsunderunchanged120incrementcompression (.9700107scale) with121physicalframedumps. Freezeinputhashes first, requireendpointparity1e-6; nooptimization,edit,calibration,parameteradjustmentorwinnerselection. Preserveeveryreplay/failure. TwoexistingthermalGIFsusefixedseed786, recipeA, lowestIDpositive-to-negativemeanexampleat20K(234)and30K(262), explicitlyillustrationsselectedforpresentation; full30statsremain. Thermalsourcesalreadyhave200actualframes; no newthermalsimulations. SevenGIFs4.000seconds each, allactual121/200frames, periodicbondrendering,fixedmatchedaxes,nointerpolationormotionexaggeration. Exactsource/graph/trajectory/helper/LMP/codehashes andPBSjobsretainedunderadam_showcase.


Adam showcase trajectory jobs completed: Reid93/84/96/97/52 jobs4676271–4676275, existingthermalGIF renderjobs4676276–4676277, collection4676278. All10/10frozenReidreplayendpoints match priorphysicalresults withmaxabsoluteerror5.551115123125783e-17;121/121frames each. BoththermalGIFsourcesretain200/200frames. All7renderedGIFs use4.000seconds. Initialcomputehost qsub wrapperreturnedexplanatorytextwithzeroexitstatus, nowpreserved as submission_rejected_compute_host.json; job-IDregexvalidationadded andactualjobswere submittedthroughSSHloginserver. No automaticapprovalrejection or scientificfailure; no duplicateLAMMPS jobs fromrejectedattempt.

User-requestedengineeringnotebooks01–05 deletedandreplacedwith01_adam_engineering.ipynb andsmallread-onlypresentationhelper. Currentdocslinksupdated;historicalresultdirectories/logs/recipes retained. Compactstatisticsincludeall10Reid/all30thermalnetworks, positive→negative denominators10/15 and3-seedthermalmeans/SD. AllReidnonconversionsremainvisible. Notebookhasonepairedstatisticsfigureand7embeddedGIFs, nooptimizer/positions-onlycomparison. GIFpalette/resolutioncompactionchangesonlypresentation, preservesphysicalframesand4splayback. Finalasset/nbhashes capturedinshowcasecompletionmetadata.


### Repository organization and single-network engineering verification — 2026-09-09

Active code is now grouped under `src/lss/dynamics/` (AE/propagator architectures, training, post-fit evaluation and analysis) and `src/lss/engineering/` (frozen model loading, geometric edits, joint Adam, final verification and animation). Retired CV/full-space, peptide, Hessian, matched-GNN and older inverse-design implementations moved to `src/lss/past_experiments/`. Small legacy import aliases preserve historical checkpoint loading. Notebook groups are `notebooks/dynamics/`, `notebooks/engineering/`, and `notebooks/past_experiments/`; dated logs and frozen result snapshots retain their historical paths. See `docs/project_structure.md`.

The shared propagator training default was corrected from response-R² selection to validation loss. Cache version 4 includes the default checkpoint metric so old implicit response-selected caches cannot match the new default. Existing explicitly configured historical recipes and frozen snapshots are unchanged. The full runnable suite passed 64 tests and six subtests; three tests depending on an absent external historical script were archived with their source and explanation, while the independent old-model test remains runnable.

`notebooks/engineering/02_engineer_one_network.ipynb` executes pristine Reid 84 through the local frozen AE/readout, joint Adam, saved final graph hashes, final-only athermal LAMMPS compression, and a comparison GIF. This is a known development-case refactor verification, not a new held-out reliability experiment. Recipe: learning rate 0.08, log-stiffness bound 0.25, patience 50, maximum 1200 steps, min_delta 1e-5, position radius 4, blocking-node masking; strict lowest feasible ML score including the original. The frozen model/readout comes from `reid_positive10/study_v1/prepared/reid/prepare/prepared.pt`; the source is raw `data/reid_200_frames.pt` frame zero. Optimization stopped at 594 steps with best calibrated ML score -1.0305200815200806. The downstream calibrated latent score is not an autonomous prediction of positions.

Run: `runs/engineering/reid84_known_case_20260909T065036Z`. Both declared final-only checks completed locally, with no PBS job: physical p-ratio original +0.14536385489933665, optimized -0.1892959352030621, 2/2 valid designs, 121/121 actual frames each, retained-frame versus separate endpoint difference zero. Protocol: 120 x-compression increments to scale 0.9700106999999999 with y box relaxation and the established side-displacement endpoint estimator. No thermal velocity seeds apply to this athermal check; one known network and one frozen model, so no seed-SD or unseen-network claim. Original and optimized graph hashes, dataset/bundle/source hashes, LAMMPS binary identity, protocol, logs and trajectory hashes are in the run artifacts. No simulation output affected optimizer acceptance, stopping or design selection.

The notebook embeds the optimization plot, before/after table and all-frame GIF. The GIF is also saved beside it in `notebooks/engineering/Engineer one Reid network/`; 640×336, 64 colors, 121 frames, 4,000 ms. The existing cohort notebook and all seven GIFs remain together under `notebooks/engineering/`, including the full Reid and mixed-temperature cohort statistics.

Comparison with the earlier cohort CSV is approximate, not bit-for-bit full-run parity: historical Reid 84 used 555 Adam steps, predicted score -1.0304474830627441 and physical p-ratio -0.1890635266367446. The reorganized end-to-end run used 594 steps and physical p-ratio -0.1892959352030621, a difference of -0.0002324085663174824; original p-ratios agree within 5.6e-17. The separate two-step learning-rate-0.02 extraction test is exact. The full-run discrepancy is retained rather than tuning against LAMMPS to eliminate it.
The notebook fixes Torch CPU threads to 1; the historical sweep runner fixed them to 4 (`hpc/experiments/reid_joint_sweep/run.py`). This is a concrete execution-setting difference and a possible source of floating-point optimizer divergence; causality has not been established. No physical result was used to retune the recipe.

Final notebook execution and persistence audit: the first 594-step draft did not save raw history/edits; an ML-only recovery failed exact graph parity and is recorded as a failure, not attached as the original history. An earlier background kernel retry was subsequently found complete as `runs/engineering/reid84_known_case_20260909T065259Z`; it is a separate full known-case run and is retained. Root executed the final notebook as `runs/engineering/reid84_known_case_20260909T070419Z` after adding history/edits persistence before verification. These two updated executions match byte-for-byte for history.csv and edits.pt, both stop at 565 steps with frozen ML score -1.0302908420562744, and both physically verify original +0.14536385489933665 → optimized -0.18743888736952807. The final notebook and adjacent GIF show this latest result, not the earlier draft. All three original/final pairs are retained: 6/6 valid physical checks, 121 frames each; repeated checks of one known network are not six independent engineered examples. No recipe was retuned from physical results.

All three runs now have hash-verified source snapshots. The final two preserve history and edits with hashes in the pre-verification frozen recipe. The initial kernel socket-permission failure, missing first-draft history, failed exact recovery, and interrupted duplicate ML-only recovery remain documented. Final execution request, notebook source, launcher, all-run metrics and notebook/GIF hashes are under `notebooks/results/engineering_workflow/`. Final checks: four executed notebook code cells, no error outputs, GIF 640×336 / 121 frames / 4,000 ms, exact current-run history/edit persistence, matching current-run repeats.


### Documentation and current-import cleanup — 2026-09-09

Current documentation now starts at `docs/README.md`; superseded plans, audits and historical validation moved to `docs/past_experiments/`. Engineering-goal instructions and dated scientific evidence are preserved. Log references were updated to relocated documents without changing findings.

Removed runtime module aliases (`lss.latent`, `lss.models`, retired top-level modules), the dynamics model re-export facade, legacy Box-module injection, unused plotting/geometry aliases, the no-op reference-context switch and ignored stiffness-exponent argument. Current callers import concrete dynamics or explicit past-experiment modules.

Converted a separate engineering working bundle to `models/engineering/reid_frozen_ae.pt` with `tools/migrations/checkpoint_imports.py`; historical source remains unchanged. Source SHA256 b137ccfce3991d43fff5a7a57c271f0eb87327d83028d8fb834662e6d3e754ff; converted SHA256 404dd51f938a0cd358714e2a72569d17a4fba5e9d131be2cf1d087407604024e. Only pickle import references changed; other ZIP members/tensor storage were preserved. Conversion manifest includes the tool hash and all rewritten references. Engineering notebook setup now points to this working copy, while historical outputs and manifests remain untouched. All six raw dataset pickle inventories already used current types, so data were not converted.

Validation: 69 tests passed. Converted-model Reid 84 check (two joint Adam steps, learning rate .02, bound .25, one CPU thread, remaining optimizer defaults unchanged) exactly reproduced initial score .043594829738140106 and final score .007926616817712784 without loading legacy modules. The score is the existing calibrated downstream proxy, not a physical or autonomous position prediction. `models/engineering/validation.json` records the check. No new training, LAMMPS verification, p-ratio evaluation, split change or final-test use.
