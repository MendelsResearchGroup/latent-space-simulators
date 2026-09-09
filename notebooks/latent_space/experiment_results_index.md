# Experiment results index

Research objective: compact shared mechanical-response coordinates and reliable
source-wise rollouts, including noisy-LJ and mixed-T transfer. Initial response
prediction, deformation reconstruction, and causal rollout are separate claims.
Latent dimension is open: explicitly compare 2D and 4D rather than assuming 4D.

## Read before choosing experiments

- `latent_matched_study_log.md`: current controlled studies, recipes, source-wise results, confounds, and scheduling changes.
- `06b_experiment_log.md`: historical mixed-T successes and current reconstruction attempts. Historical 0.901 mixed-T rollout used a 4D all-four AE but Reid + low-T-only delta propagator. Original checkpoint/full recipe unavailable; recent reconstructions are not exact replays.
- `07b_experiment_log.md`: historical source-wise dynamics/AE ablations, including positive mixed-T results and invalidated experiments. Read before re-testing an old idea.
- `latent_simulator_research_audit.md`: claim boundaries, historical evidence, and research gaps.

## Current machine-readable results

Paths below are relative to `notebooks/results/`.

| Study | Status at index update | Results / ledger | Key use |
|---|---|---|---|
| Matched representation screen | Complete | `latent_matched_study/` | Dimensions 2/4/6/8, shared vs separate AEs, three seeds |
| Matched dynamics and transfer | Complete | `latent_matched_rollouts/` | Frozen 8D AE; source-wise rollout failures and successes |
| AE-unseen mixed-T | 27 rollouts complete | `latent_no_mixed/sourcewise_aggregate.csv` | Excluding mixed-T from AE and propagator did not recover its rollout |
| CPU timing | Complete | `latent_full_ae_benchmark/multinode_summary.json` | Eight threads fastest observed full-data AE setting; shared-node timing can differ |
| In-memory normalization | Passed all 160,400 frames | `in_memory_normalization_audit/*.json` | Retain raw files with audited in-memory normalization |
| Motion baselines and drift | Nine complete | `latent_science_followup/diagnostics_aggregate.csv` | Learned acceleration beats LJ extrapolation; teacher forcing improves mixed-T |
| Low-data AE and probes | Six complete | `latent_science_followup/probes_aggregate.csv` | Separate 29-network/source calibration; initial information does not imply reconstruction |
| Matched horizon and AE update budget | Eighteen complete | `latent_controlled_followup/{stability,reconstruction,probes}_aggregate.csv` | Longer horizon/repeated samples did not solve all-source rollout/reconstruction |
| 06b reconstruction and dimensionality | Ten runs complete | `06b_4d_reconstruction/jobs.jsonl`, `d{dimension}_s{seed}/` | 2D mixed-T rollout 0.904 ± 0.012; 4D not necessary; LJ unresolved |

## Interpretation and operation

Read per-run `recipe.json` and `completed.json` before interpreting aggregates.
Job ledgers may include cancelled jobs and replacements: count completed run
directories, not submission lines. Preserve seed SD and source-wise valid/total
counts; do not treat repeated fixed-AE controls as independent AE seeds.

Use `mendels_q` and shared single-node placement; no explicit host pinning.
Run broader controlled matrices when justified, save all outcomes, and avoid
repeating resolved questions without identifying the new factor. Keep reserved
final-test data untouched during selection. Do not claim missing historical
artifacts have been exactly reproduced.

## Current working baseline

Reconstructed 06b 2D AE-all4 / delta-propagator-Reid+low-T gives validation
rollout R² 0.814/0.947/0.904 for Reid/low-T/mixed-T over five seeds. Preserve
it when adding LJ. This is propagator transfer, not AE-unseen mixed-T transfer.
LJ AE response reconstruction and rollout remain poor. Exact historical replay
is still unavailable, but strong mixed-T behavior is recovered on the current
validation split. See latest 06b log entry before proposing more experiments.

## Completed AE-exposure test

`06b_ae_mixed_ablation/`: five completed 2D excludes-mixed-T AE runs paired with the five
completed successful 2D includes-mixed-T baselines. Propagator always trains
only on Reid + low-T. Mixed-T is loaded only after model selection; source-wise
AE/rollout and paired contrasts confirm successful exclusion (mixed-T endpoint rollout
R² 0.868 ± 0.033) versus inclusion (0.904 ± 0.012). See 06b log for sample-budget and RNG
qualifications. AE exposure is not necessary for strong endpoint transfer in this recipe;
early-time performance and noisy-LJ remain unresolved.

## Active priority: repair noisy-LJ AE before dynamics

User explicitly requires adequate LJ reconstruction first. The successful
mixed-T-unseen AE's LJ frame-100 response R² is -0.011 ± 0.342; it is not fixed.
`lj_ae_repair/`: 40 AE-only runs, eight supported recipes × five seeds, covering
longer training, equal-source objective/selection, response-aware checkpoint
selection, width, 2D/4D, and LJ edge information. Mixed-T remains unavailable
until after fitting/selection. Full matrix and comparison qualifications are
in the latest 06b log entry. Preserve the successful transfer baselines and
advance to LJ propagator supervision only after representation evidence.

Automatic collection job `4670235.zeus-master` waits with `afterany` dependencies
on all 40 AE-study jobs. It writes aggregates, incomplete-run status and
`lj_ae_repair/review_packet.md`, plus a collection entry in the 06b log. This
schedules result collection even if some runs fail; it does not reopen the
assistant conversation or launch further training autonomously.

Partial AE review: 28/40 completed; three full five-seed recipes still have weak LJ response reconstruction (best of these: 0.210 ± 0.285). Saved `lj_ae_repair/prediction_diagnostics*` quantify bias, correlation, response spread and relative strain errors at all six frames. See latest 06b log. Strain-aware training is the next hypothesis; no LJ dynamics launched.

## Dynamics-only requirement and matched 6D/8D capacity study

User clarified the project-wide scientific requirement: learn from network
states/trajectories and structure, never add p-ratio, strain, or expert-derived
supervision. Expert observables remain post-training diagnostics, not training
inputs, objectives, or checkpoint/model-selection criteria. This supersedes the
previous proposed strain-aware objective; no such objective was implemented.
Historical response-selected AE and propagator runs remain recorded, but are
not compliant evidence of fully observable-blind learning. Future propagator
baselines also need state/trajectory-based selection.

New AE capacity matrix: dimension6 and dimension8, each seeds
3456456/123/456/786/2026. Matched against equal_source (2D) and dimension4:
width96, 32 tokens, four stored edge channels, normalized displacement input
and target, equal-source/graph MSE, worst-source validation reconstruction
checkpoint, 40 epochs/patience8, batch32, lr1e-4, weight decay1e-5, frames0–100,
20 train/20 validation each Reid/low-T/LJ. Mixed-T excluded until post-fit
validation; final test untouched. Disable the diagnostic per-epoch p-ratio
callback; it never selected checkpoints in the matched 2D/4D controls.
No additional expert features or LJ edge augmentation. No propagator training.
Prior 8D evidence used different width/edges/splits/source exposure and cannot
substitute for this matched comparison. Dimensional capacity remains a hypothesis.

Recipes/results/job ledger: `notebooks/results/lj_ae_capacity/`; frozen source
and hashes in `code_v1`. Runner/submitter/collector:
`scripts/{run,submit,aggregate}_lj_ae_capacity.py`. Configuration assertions
passed for all ten variants/seeds; shared mendels_q, 8 CPUs/32GB, no host pins.
Existing and new collectors rank only state reconstruction among eligible
non-response-selected recipes; p-ratio/strain tables remain diagnostic only.

Architecture research and overnight expansion delegated to Terra (`terra_shared_ae`) at user request. Durable instructions and primary-source hypotheses: `docs/past_experiments/research/shared_dynamics_architecture_brief.md`. Priorities: nonlinear neighbor message passing, pooling/capacity controls, followed conditionally by learned temporal state and rollout robustness. Broad queued AE experiments authorized after smoke checks; submission/results will be recorded by the executing agent.

Deeper architecture audit: `docs/past_experiments/research/shared_dynamics_deeper_audit.md`. Saved history diagnostics suggest LJ training error is also high (~0.30 normalized MSE vs ~0.33 validation); no irreducible-noise claim. Identified nonzero-mean standardized-edge reversal inconsistency in frozen model; algebraic check `scripts/audit_ae_edge_reversal.py`, downstream impact pending. Terra notified to isolate correction from architecture changes. Proposed encoder-vs-decoder latent optimization, latent-use controls, and predictive temporal objectives without expert supervision.

Consolidated execution plan: `docs/past_experiments/research/shared_dynamics_execution_plan.md`, assigned to Terra at user request. Stages A–F cover correctness controls, information-bottleneck diagnostics, broad spatial architectures, temporal representations/history, source interference and rollout robustness. Early independent work can queue after smoke checks; later stages depend on recorded state-based evidence.

## Completed AE collection and pipeline audit — 2026-09-07

All40 lj_ae_repair and all10 lj_ae_capacity runs completed; aggregates refreshed.
Matched LJ endpoint reconstruction diagnostic R² mean±SD: D2 -.202±1.435,
D4 .092±.235, D6 -.089±.391, D8 -.170±.620; five seeds and20/20 valid each.
Position MSE remains approximately8–9e-6. Full source-wise coordinate and
response summary: lj_ae_capacity/matched_dimension_summary.csv. More dimensions
alone have not resolved this recipe; no conclusion of dimensional impossibility.
All use historical uncorrected edge orientation; 6/8 disable diagnostic callback
while2/4 had callback without response-based selection. Expert metrics are
post-fit diagnostics only, not model-selection criteria.

Pipeline audit saved at docs/past_experiments/research/latent_pipeline_audit.md. Default cache
configuration matching is now enabled and regression-tested; current study
force-training means stale reuse does not explain its outcomes. Targeted active
pipeline suite:17 passed plus6 subtests, including batch-vs-single parity,
normalization inversion and future-data independence at fixed latent. Three
separate legacy simulator tests failed on stale argument signatures; recorded
rather than hidden. Edge-reversal corrected controls and architecture study
remain delegated to Terra, with smoke failures and retries preserved.

User reports working LJ AE elsewhere. Historical08/09 comparison saved in docs/past_experiments/research/lj_historical_positive_control.md:08 LJ R2 .6296 on30 vs .4681 on110;09 .468 on110 under ordinary reconstruction selection. Recent6D differs materially in data budget,width,decoder tokens,objective,edges and evaluation split; it was not a positive-control replay. Exact user example requested; checkpoint not found in current08/09 results directory inventory.

## Notebook08-inspired sweep and approved reserve migration

User approved moving30 reserved LJ networks into training. New authoritative
reserve manifest: notebooks/results/lj_ae_08_bridge/split_manifest.json; preserves
original historical manifest/hash and exact moved IDs. Training30/30/60 for
Reid/lowT/LJ, validation20 each; LJ reserve120 instead of150. Cross-source
node-labelled topology overlap checked; no other reserved IDs needed exclusion.
Never evaluate the former150 LJ reserve as an untouched final test after this
study. Mixed-T remains excluded fitting/selection.

40 runs: near08 anchor (D6,width192,tokens46,150frames) and isolated variants
D2,D4,D8,width128,tokens32,101frames,source_mean, each five seeds
3456456/123/456/786/2026. Anchor uses mean objective,50epochs/patience8,batch32,
lr1e-4,wd1e-5,5edge channels plus historical LJ distance3 augmentation.
All select ordinary validation reconstruction, no response callback. Historical
frozen model orientation retained to isolate recipe; corrected architectures
run separately under Terra. Not exact08 replay: seeds,validation split and
selection differ. Frame150 diagnostics extrapolate one beyond150frames0–149;
frame149 included explicitly. Broader evaluation5/10/25/50/75/100/125/149/150.

Eight expanded recipes passed assertions. Scripts run/submit/aggregate_lj_ae_08_bridge.py;
frozen code_v1/hashes,intended matrix,recipes and job ledger under lj_ae_08_bridge.
Submitter has an exclusive lock to prevent overlapping launcher duplicates.
Shared mendels_q8CPU32GB24h, no host pins. Collector afterany records incompletes.

## Near08 sweep submission complete

All40 intended variant/seed cells submitted. Earliest jobs show successful
training with30/30/60 trajectories and4500/4500/9000 training frames for the
150-frame recipes. Interruption stranded one accepted job outside the ledger:
tokens32 seed123 original4670689, recovered via recipe.json. Duplicate4670700
terminated on FileExistsError without overwriting/training; qdel found it already
terminal. Both attempts preserved in jobs/recovered_jobs and job logs. Collector
dependencies include original4670689, so it cannot collect before that run ends.
Submission now checks existing recipe records on resume as well as holding a
launcher lock. Matrix aggregation counts completed run directories, not attempts.
Exact collector ID and dependencies: lj_ae_08_bridge/collection_job.json.

Memory request update: future ordinary attention-AE jobs in lj_ae_08_bridge.pbs, lj_ae_capacity.pbs and lj_ae_repair.pbs now request16GB instead of32GB, retaining8CPUs/shared placement. User authorized resource reduction. Evidence: lj_ae_08_bridge/resource_snapshot.json (median6.7GiB,max7.2GiB observed during training; full validation peak not established). Current running allocations unchanged. Message-passing/optimization diagnostics keep their separate allocation until measured.

## New downstream network-design experiment

notebooks/network_design/01_latent_guided_stiffness_design.ipynb plus design.py,
run.py and shared8CPU16GB run.pbs. New reconstruction-selected2D AE trains
Reid/lowT20each with20validation each; no response labels enter AE training.
Source-specific downstream calibration first10validation networks, design pool
remaining10, selecting3high-response examples. FreezeAE; bounded log-stiffness
edits±.25, fixedgeometry/topology, zeros unchanged; opposite and5matched-norm
random controls. Current example defaultsReid. Not a replay of06; calibration
and design observations are distinct but both were AE validation, not finaltest.

Finite-difference gradients and bounds/export tests passed (2tests). Notebook
schema/code compilation passed. MetaForge sibling identified (user typed
MetaForce), physicalvalidation integrated via auxetic.run_elastic_simulation.
All locatedLAMMPS binaries currently fail libpng15.so.15 dependency, including
cluster login probe. Physical failures recorded explicitly; no claimed reduction.
Elastic-moduli p-ratio protocol is distinct from frame100 trajectory calibration;
compare each edited graph against its freshly simulated original.


### Network-design execution verification, 2026-09-07

Supersedes the LAMMPS blocker above: project-local pinned LAMMPS2023.8.2.3.1
with mpi4py-mpich3.1.5 runs successfully. Full tiny-training notebook smoke
completed 8/8 Reid physical evaluations; five gradient/export/pipeline tests
passed. Exact recipes, IDs, checkpoint, controls, CSVs and execution log:
`notebooks/results/network_design/smoke_reid_lowT_2d_s786/`.
This is an execution check (1 AE epoch, 2 training networks/source, 3 design
steps), not an efficacy experiment. Reid index2 original elastic p=0.29082325,
saved trajectory frame100=0.22551994 and final=0.22388364; no registry value on
this graph. MetaForge averaged-moduli response and boundary-side displacement
response are different protocols; baseline agreement remains unresolved.
Designed elastic change=-2.4e-7 despite improved latent readout: negative result,
not successful engineering. Calibration/design are validation subdivisions;
final test untouched. Preserved failed attempts include missing config field
in PBS4670942 and two legacy-package import collisions in smoke attempts.
Corrected full 40-epoch/patience8 recipe submitted PBS4670957, seed786,
8CPU16GB mendels_q free:shared; frozen source and hashes in network_design/code_v1,
external MetaForge hashes and job records beside it. Await full results.

### Network-design endpoint baseline audit v3, 2026-09-07

Validation-only endpoint audit complete: v2 PBS4670977 (Reid, maxRSS862808kB)
and PBS4670978 (dePablo low-T,1046364kB); unwrapped-coordinate v3 PBS4670982
(Reid,878196kB) and PBS4670983 (dePablo low-T,1033524kB). Each completed5/5.
Protocol: athermal quasistatic x-compression to the saved final x scale0.9700107,
y stress zero, fixed shear,30 minimization increments, with first/final metrics
only. V3 same-boundary side-ratio MAE/max: Reid0.021912/0.043951; dePablo
low-T0.001931/0.005630. DePablo index191 confirms the material wrapped-coordinate
artifact: wrapped error0.205940 becomes unwrapped error0.000376 (one saved final
node outside the box). Reid retains finite residuals up to0.043951; the original
saved trajectories are finite-rate and this replay is quasistatic, which may
contribute but has not been isolated as their cause. Inputs,
per-index values, box-nu errors and protocol-context elastic means:
`network_design/baseline_audit_v1/terra_endpoint_summary.{json,md}`; raw v3
tables: `network_design/baseline_audit_v1/{reid,depablo_low_temp}/endpoint_compression_v3/`.


### Endpoint network engineering queued, 2026-09-07

Uses the completed 5+5 baseline audit above. Frozen reconstruction-only 2D AE
PBS4670957 is the prerequisite. New helper reproduction PBS4670994 and training
must succeed before Reid PBS4670995 / dePablo PBS4670996 start. Each source uses
10 validation networks for downstream final-response ridge calibration, then
selects 3 high-response networks from the other10. Exact IDs will be written by
the notebook before physics. Labels never update or select the AE. Not finaltest.
150 bounded log-stiffness steps ±0.25, seed786+index; 5 matched random controls,
opposite direction, uniform scaling control and unchanged graph:27 simulations
per source. Validated2D compression, matched final x scale, y stresszero/fixed
shear; initial/final unwrapped response only. Readout validation against the
constant mean is also saved. Source-wise physical differences, random-control
comparisons, negative results and failures collected automatically. Recipes and
job/dependency records: network_design/engineering_endpoint_v1/submission_recipe.json;
frozen source hashes: network_design/code_endpoint_v1/hashes.json.
Pending, no engineering success claimed. Each design job8CPU16GB mendels_q shared.

### Endpoint network engineering completed, 2026-09-07

Exploratory validation-only endpoint engineering completed after AE4670957 and
helper4670994: Reid4670995 and dePablo low-T4670996 each completed3/3 candidates
at seed786. Final first-to-last unwrapped boundary-ratio designed-minus-original
changes: Reid indices2/53/59 = -0.001548/-0.002052/-0.003162, with2/5,3/5,5/5
matched random controls beaten; dePablo low-T indices70/78/96 =
-0.002369/+0.000339/+0.009402, with3/5,2/5,0/5 controls beaten. Uniform-scale
changes are near zero (absolute maximum1.14e-06). Same final endpoint protocol:
2D quasistatic compression to matched x scale, y stresszero, fixed shear,
initial/final unwrapped boundary ratio. One-seed validation subdivisions do not
establish reliable engineering. Exact recipe, code-manifest hash, candidate rows
and result paths: `network_engineering_log.md` and
`network_design/engineering_endpoint_v1/{engineering_summary.json,submission_recipe.json}`.


### User-authorized model simplification matrix — implementation/smoke stage

`reference_simplification_log.md` records prior context failures and the matched
new recipe: per-node reference96/16/8 × latent2/4 × five seeds, each with paired
propagator graph-context16/none on the exact same frozen AE. Reid+lowT20train/
20validation each; mixedT20validation only after both checkpoint selections;
no final test. AE reconstruction and propagator latent-delta loss select models,
no expert-observable selection.30 AE recipes /60 paired dynamics fits, with
eligible existing R96/D2/seed786 AE reuse. New smaller-reference model classes,
correct raw context sizing and bundle reload;10 targeted tests passed. Smoke
jobs4671230/31/32 test baseline and both reduced widths before the full matrix.
Artifacts and hashes: notebooks/results/reference_simplification/. Full status
and results are recorded in the dedicated log, ledger and source-wise collector.

### Source-specific latent direction and engineering follow-up — 2026-09-07

`network_engineering_log.md` records the signed-direction audit, pilot4671363,
and new calibration→selection→optimization dependency chain. Four separate
calibration networks/source compare z0/z1/PC1/ridge2D and both edit signs before
six disjoint design candidates. Four equal-budget optimization arms include
fixed source sign and random directions; first/last physics and finer endpoint
confirmation. Exact556-evaluation plan, frozen code and job ledger:
`network_design/calibrated_direction_v1/`. Exploratory validation, AE frozen,
no final test; no optimized/control-confirmed success claimed yet.

### Local-gradient direction pilot completed — 2026-09-07

Exploratory validation-design pilot PBS4671363 (dePablo low-T index96, seed786)
completed44/44 physical evaluations. Frozen-AE local-gradient versus
finite-difference check relative error0.000170. With equal budgets (four rounds
of four proposals per arm), 60-step endpoint confirmation gave AE-gradient
p0.189101→0.183233 (change -0.005868) and bidirectional random
0.189101→0.187210 (change -0.001891): additional reduction0.003977, 3.10×
random's absolute reduction. The candidate is a validation subdivision and final
test remains untouched; one candidate/seed does not establish reliable
engineering. First/last unwrapped boundary response under 2D quasistatic matched
compression is the physical metric, with p-ratio used only downstream. Full
machine-readable summary and provenance:
`network_design/direction_v1/terra_summary.json`; raw final confirmation:
`network_design/direction_v1/depablo_low_temp_96_s786/summary.csv`.

### Corrected full-span simplification sweep submitted — 2026-09-07

All three code_v2 smoke jobs4671316/17/18 passed, including both context modes
and saved-bundle reload.23 targeted/regression tests passed. Explicit full
configuration check covers AE frames0–199 and all199 propagator transitions.
The corrected30 AE /60 paired-propagator recipe matrix is now fully submitted,
with afterany collector4671446. All use mendels_q8CPU16GB shared, no host pins.
Seeds3456456/123/456/786/2026 × retained reference96/16/8 × latent2/4.
No reuse of the101-prefix engineering AE; all matched full-span AEs train fresh.
Exact jobs/dependencies: `reference_simplification/jobs.jsonl` (code_v2 non-smoke).
Preserve code_v1 cancellations and earlier smoke results as separate attempts.

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

### Geometry plus stiffness target -0.1 — implementation and smoke stage

`network_engineering_log.md` records the matched18-cell continuation matrix:
Reid2/53/59 + low-T70/78/96 × stiffness / positions / joint, same best prior
stiffness starts, frozenAE, validated differentiable physical/reference geometry.
Fixed box/topology/measurement boundaries; updated rest lengths; physical-only
edit acceptance and60/120-increment target confirmations. Four candidate processes
per8CPU shared job. Geometry checks passed; smoke runs precede full submission.
Artifacts `network_design/geometry_target_v1/` and `geometry_smoke_v1/`.

### Geometry target matrix and simplification sweep — 2026-09-07

18 geometry/stiffness jobs submitted with afterany collector4672383; early Reid53
stiffness and joint designs independently confirmed below-0.1. Exact job IDs and
provenance: network_engineering_log.md and network_design/geometry_target_v1/.
Remaining engineering outcomes pending; do not claim cross-network target success.

Reduced-reference code_v2: all30AE/60propagators complete across five seeds,
270 source rows,100/100 evaluations per source/config (20 unique networks).
Collector failure repaired and preserved. See reference_simplification_log.md and
reference_simplification/collection_code_v2/. Removing16D graph context hurts
every paired source/seed coordinate comparison; retain it. Per-node8D or16D
reference with4D latent has near-baseline endpoint rollout MSE across sources,
whereas2D with narrow reference costs about15–16% on low-T. No formal equivalence
or speedup established; no LJ or final test here. Mixed-T was unseen in both fits.
Keep96-reference/2D-latent control and investigate8-reference/4D-latent across full
trajectories before a default change. Expert response metrics remain post-fit only.

### LJ evidence reviewed for16D node-reference continuation — 2026-09-07

User prefers16D node reference and accepts a modest low-T loss. Keep16D graph
context; latent2/4/6/8 remains an empirical question. Completed bridge40/40 does
not establish robust LJ reconstruction: at last trained frame149, D2/4/6/8 LJ
MSE is2.094/2.108/2.130/2.084e-5 (five seeds;100/100 evaluations each), despite
near08 width/data/token budget. No bridge propagator was trained. Full-prefix
coverage and source-specific synthetic LJ edges confound comparison to the
compact-reference sweep. Shared architecture study30/40 complete; MP2/D2 has
five seeds and improves corrected-control coordinates, but LJ frame100 remains
8.252e-6. Ten uncompleted cells are directory-collision failures, not model
failures. Exact evidence, caveats and proposed conditional experiments:
`docs/past_experiments/research/compact_reference_lj_next_plan.md`. Plan only, no new submissions.
Prioritize matched shared/LJ-only reconstruction diagnostics with16D reference
and wide control, full-time coverage, then conditional propagator training.

User clarification: next noisy-LJ experiments MUST retain LJ-edge augmentation
through original spring-graph distance3 (new distance2/3 pairs, raw stiffness0,
relation indicator1). Shared five-channel edge schema; consistent train/val/inference.
Plan and AGENTS.md updated. Older four-channel architecture runs lack this
required representation and cannot settle LJ performance with the intended edges.

### Compact-reference LJ reconstruction execution prepared — 2026-09-07

User authorized continued reconstruction work and a motion/data audit. Exact
50-run first matrix and limitations: docs/past_experiments/research/compact_reference_lj_next_plan.md.
Mandatory LJ graph-distance3 edges (raw added stiffness0), five-channel schema,
full200-frame span, reference16 priority; latent2/4/6/8 + wider controls +MP2
with required edges + LJ-only reconstructibility controls; five seeds. No new
propagators until reconstruction evidence. LJ-only normalization/exposure differs;
do not claim isolated causal interference.21 active tests pass; five pre-existing
legacy tests reproduced failing in unchanged snapshot and recorded separately.
Three smoke jobs precede full sweep submission. Motion audit4672395 running.
Results/provenance: compact_lj_reconstruction/ and compact_lj_motion_audit/.

### First LJ motion and fitting evidence — 2026-09-07

Motion audit4672395 completed;10/10training networks/source,200frames each,
common5-channel schema with LJ distance3 augmentation. Fixed first10bridgeTRAIN
IDs, saved under compact_lj_motion_audit/. Coordinate RMS displacement fromframe0
mean: Reid.006855, low-T.006627, LJ.006741. Increment cosine atlag1 is>.9996 for
all three; atlag10 Reid.999082/low-T.999306/LJ.995366. This sampled LJ data is
smooth and similar in amplitude; no basis for assuming rapid irreducible noise.
No raw-versus-minimum-image increment differences in these30trajectories.

Best affine-map residual RMS mean: Reid.002111,low-T.001563,LJ.002847. Per-network
centered in-sample PCA residual fractions rank2/4/8: Reid3.538e-5/3.746e-7/2.182e-8;
low-T2.599e-6/7.427e-8/1.411e-8; LJ9.753e-4/5.216e-5/2.198e-6. These fit a separate
basis using all observed frames of each training network: optimistic descriptive
compression, NOT a sharedAE result or a forecast/generalization bound. Evidence
suggests learning network-specific spatial deformation deserves attention; a
large evolving latent is not the only candidate explanation. Per-frame follow-on
will compare AE errors to motion/affine baselines on matching training identities.

Historical gap audit: near08 five-seed selected-epoch LJ normalized train/val
reconstruction .31959±.01067/.33543±.00751; source_mean .32681±.00865/.33179±.00287.
Training numbers are online epoch averages at the validation-selected epoch,
NOT frozen-checkpoint reevaluations (historical_training_gap metadata). LJ error
was high even in fitting; this alone neither proves interference nor rules out
insufficient training. Metrics are coordinate-only, no mixed-T/p selection.

First compactLJ smokes4672416/17/18 failed before training on missing required
cfg early_stop_min_delta. All rawLJ preflights passed (2032added edges on selected
train graph, stiffness0/originaledgespreserved). code_v1 frozen and failure
artifacts preserved; runner correction and new immutable retry version required.

### Per-frame motion audit and review notebook

Follow-on4672421 completed exit0,2m20s,PBS peak5.55GB.6000 frame-level rows
(30training networks×200), persistence valid5970/6000 and extrapolation5940/6000.
Original audit4672395 and code preserved; frame_rows_v1.json and metadata are
separate artifacts. Atframe100, training-set mean best-affine coordinate MSE is
about6e-6 Reid,3e-6 low-T,8e-6 LJ. That LJ value is of similar order to historical
AE validation MSE, but samples differ; this is a hypothesis about missing spatial
deformation, not a matched performance comparison. Incoming AE train diagnostics
share the first5training IDs with this audit, enabling a valid matched residual
comparison. Affine/PCA fits use observed targets and are NOT forecasts or losses.

Review notebook: notebooks/latent_space/diagnostics/compact_lj_reconstruction.ipynb,
executed successfully with inline Editorial plots, no figure exports. Shows
training-motion summaries, per-network in-sample PCA, per-frame fitted baselines,
and any completed compact AE source/frame results. Partial runs remain labeled.

### Full compactLJ sweep queued — 2026-09-07

Corrected code_v2 smokes4672423/24/25 completed (213.6–222.4s, maxRSS5.63–5.65millionkB).
All3 fitted, produced source/frame controls and passed latent/decode bundle reload;
LJ original spring preservation,2032 added zero-stiffness relations on preflight
train graph, indicator schema and idempotence verified. Preserve failedcode_v1.
Training snapshot SHA256: 3fbd7c8d6e117ce462ef290fabef3592c2b786bb45a6b8325d4fb9f353eacfb9.

All50 unique variant/seed cells submitted from immutablecode_v2, job range
4672434.zeus-master through 4672486.zeus-master (not necessarily contiguous).
Exact mapping and hashes: compact_lj_reconstruction/jobs.jsonl; intended recipe
written before qsub. Shared mendels_q8CPU16GB24h, no pins. First full shared run
confirms30Reid/30low-T/60LJ and6000/6000/12000 training frames; first LJ-only run
confirms60LJ and12000frames. No final test. No full-run failures at this check;
scientific reconstruction outcomes still pending.

Afterany collector4672487.zeus-master depends on all50. Post-fit comparison job
4672488.zeus-master depends on collector,1CPU2GB15m shared. Separate immutable
analysis_code_v1 SHA256 6ec9ef83909f52142adbf47f7c5c071a3c4df85e5e28e77552c2a8950e78a89a.
It saves AE-versus-motion fits on exactly matching training IDs and paired
shared/LJ-only coordinate differences; caveats about optimistic target-fitted
baselines and unequal normalization/exposure are persisted. No automatic expert
selection or architecture promotion. Analysis code passed a matched synthetic
train/validation, incomplete-matrix execution check. All results remain reusable.

### Completed compact reconstruction sweep and decoder follow-up — 2026-09-08

All50code_v2 fits completed; collector4672487 and analysis4672488 produced complete
source/frame tables.5seeds×20held-out networks/source=100evaluations per cell;
training diagnostics5networks×5seeds=25, not independent100networks. LJ-only did
not rescue reconstruction. Atframe199 sharedR16D4 LJ MSE3.94145e-5±1.80e-7 vs
LJ-only4.08065e-5±6.13e-7; sharedD8 4.01114e-5±1.14e-6 vs LJ-only4.25195e-5±2.30e-6.
Shared is descriptively3.4%/5.7% better; fitted normalizers and update/checkpoint
exposure differ, so this is not an isolated causal interference test. No evidence
here that mixing is the primary reason LJ fails. Increasing z2→4→6→8 or reference
16→96 gives little LJ gain. MP2 modestly improves coordinates on all retained
sources and mixedT transfer, but residual LJ deformation remains substantial.

MatchedTRAIN LJ frame199 AE/best-affine MSE: attentionR16D4 1.030, LJonlyD4 1.108,
MP2D4 .979,MP2D8 .972. Swapping same-source/same-time codes increases error, so
latents are not entirely ignored; however, AE residual remains near affine-fit
scale. Both affine and PCA fitted baselines use observed targets and are strictly
diagnostic. PCA4 uses a different spatial basis AND a time-mean field fitted from
all200frames per network; it does not show that one shared4D AE/propagator works.
Its low rank supports testing compact codes, not assuming encoder/decoder adequacy.

Exact full grid, counts, seed SDs, recipes and checkpoint/exposure summaries:
compact_lj_reconstruction/{collection_code_v2,comparisons_code_v2,review_20260908}.
Shared24kframes/750batches per epoch vs LJ-only12k/375; selected epochs typically
6–11 shared vs33–49 LJonly. These do not equate optimization budgets.

Authorized next isolated architecture hypothesis: keep corrected attention
encoder and retained16D node reference, replace independent-node decoder with
2graph-isolated self-attention layers (hidden96,4heads,FF192,GELU,dropout0,
norm_firstTrue) after combining each node reference with latent-conditioned tokens.
Only transient decoder activations are96D; no96D stored bypass. No new expert
inputs/losses.20runs: sharedD2/4/8 andLJ-onlyD4 ×samefive seeds, matched previous
full-horizon/data/training recipe. Existing compact controls reused, not retrained.
11active model/pipeline tests pass including node permutation, graph isolation,
gradients and reload. Two integration smokes gate broad submission.

Parallel frozen-code diagnostics4673318/19/20: seeds123/456/786, sharedR16D4 and
LJ-onlyR16D4, first3TRAIN LJ IDs178/199/42,frames25/100/199. Optimize z only against
observed coordinates for200Adam steps from encoded and perturbed encoded codes;
AE weights fixed. This tests local decoder expressiveness versus encoded inference,
not a new forecast or an expert training target. Artifacts compact_lj_fitted_codes/.

### Frozen-code and spatial-residual diagnostics completed — 2026-09-08

Fitted-code jobs4673318/19/20 completed exit0, AEweight hashes unchanged. Three
training networks×three seeds permodel/frame=9evaluations (not9unique networks),
frames25/100/199. Best of two200step local Adam starts. SharedR16D4 mean encoded→
optimized MSE at100:8.033e-6→7.214e-6; at199:2.909e-5→2.622e-5. Mean per-case
relative reduction about10.5%/10.3%; early25 gain19.1%. LJ-only early25 gain43.5%,
later100/19914.1%/10.2%. Local optimization does not establish globaldecoder
limits, but does not recover the missing late deformation. Exact matched199
best-affine MSE2.683e-5 on these SAME3training IDs; optimized shared decoder
remains near affine residual scale. Artifacts compact_lj_fitted_codes/{s*/rows_v2.json,
best_start_rows.csv,source_frame_summary.csv,matched_optimistic_baselines.csv}.
rows_v2 corrects a start-specific field label without rerunning; original retained.

Spatial residual audit4673360 completed exit0.675 model/seed/network/frame
observations. Each variant/frame has60validation observations=20graphs×3seeds,
and15training observations=5graphs×3seeds. Orthogonal affine +non-affine error
decomposition passes every row. At199 validation, non-affine error accounts for
93.5% sharedR16D4,94.2%MP2R16D8,90.0%LJ-onlyR16D4. Predicted/true non-affine energy
ratios (ratio ofmeanenergies) .111/.141/.123; mean residual cosine .283/.316/.290.
Thus late error concentrates in underestimated and weakly aligned local motion;
this is more specific than calling LJ noisy. No such residual is a training loss
or checkpoint metric. Exact per-model/split/frame counts and outcomes:
compact_lj_spatial_residuals/{rows.json,variant_split_frame_summary.csv,recipe.json}.

Additional code inspection/algebraic check: the current reference encoder is an
affine projection of2position features +5mean incident-edge features on connected
nodes; 16 or96 output channels do not imply that many independent structural
features. Synthetic double-precision check has centeredrank7 and affine identity
error8.88e-16. Isolation mask caveat retained. This explains why merely widening
reference may add no information; it does not prove sufficiency/insufficiency for
these datasets. reference_information_audit.json records seed/codehashes. If
spatial decoding remains weak, nonlinear topology-aware reference encoding is a
separate motivated hypothesis; not yet launched or folded into currentcomparison.

### Spatial decoder matrix submitted — 2026-09-08

Both integration smokes passed:4673321 shared196.74s,4673322 LJ-only77.51s,
maxRSS5.64GB/4.46GB in decimal kB conversion. RequiredLJedges,source scopes,
finite diagnostics and exact checkpoint reloadverified. Immutablecode_v1 SHA256
42802bfb7aa67402172555b8116f0c85af617b9f79450e87c4cc80a36a41d88e.
All20 unique cells submitted4673337–4673356 with afteranycollector4673357.
Shared mendels_q8CPU16GB24h, no hostpin. Firstfull job measured6,327,228kB.
Results compact_lj_spatial_decoder/. Existing50baselinefits reused; no repeated
baseline training. Newdecoder effects pending.11active tests passed. Notebook11
refreshed with50complete baselinefits and inline frozen-code/PCA/affine diagnostics.

### Live scheduler verification after standalone migration — 2026-09-08

At 09:08:15 UTC (12:08:15 Israel), PBS reported 18/20 full spatial-decoder
runs finished with exit status 0; LJ-only D4 seeds456/786 (4673354/4673355)
were still running at approximately3h09m, using about4.5GiB each. All15 shared
D2/D4/D8 runs finished. Collector4673357 has a system dependency hold awaiting
only those two remaining jobs; this is expected afterany behavior.
Jobs still write under `/rg/mendels_prj/alexander.z/DL-course-project/`, so
the standalone repository's copied per-run status files are stale. No scientific
results synchronized or aggregated in this check. Exact job identities, states,
exit codes and resource measurements: `compact_lj_spatial_decoder/live_cluster_status_20260908.json`.

### Notebook cleanup and renumbering — 2026-09-08

Main notebook06 is now `01_mixed_dataset_shared_latent_space.ipynb`,06b is
`02_mixed_dataset_shared_latent_rollout.ipynb`, and09 is
`03_four_source_standard_ae_pca.ipynb`. Notebook11 is now the unnumbered
`notebooks/latent_space/diagnostics/compact_lj_reconstruction.ipynb` reader.
All four retained notebook contents are unchanged. User requested deletion of
the archived notebooks: legacy04,06c,both06d,07,07b,08,10a/10b; recover them
from Git history when needed. Results, exact recipes, historical experiment IDs,
logs and model implementations retain their paths. The notebook map records the
07b seed/cache discrepancy and10a/10b epoch-budget mismatch. No new scientific
results or training were produced by cleanup.

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

### Primary success metric and frozen p-ratio evaluation — 2026-09-08

User clarified that post-fit p-ratio R² is the primary success metric, with
coordinate reconstruction error secondary and potentially acceptable for a
compact latent. This supersedes any implication that coordinate MSE alone
justifies rejecting the current AEs. AGENTS.md, CONTRIBUTING.md, README.md and
research status now state this explicitly. Training remains positions/dynamics/
structure-only, with state-based checkpoint selection and no response inputs,
losses or checkpoint selection.

Authorized backfill: all50 compact_lj_reconstruction/code_v2 and20 spatial
code_v1 frozen AEs, samefive seeds, exact20-network/source validation cohorts,
frames25/50/100/150/199. Shared models additionally evaluate the existing
mixed-T post-fit cohort. No final test and no propagator claim. Primary estimator
declared before inspecting results: physical-coordinate endpoint directional-side
p-ratio via graph_utils.calc_p_ratio_rollout_sides, same original frame0 side
groups for true and decoded positions. The same endpoint formula is also
reported in model coordinates for historical comparison: older bridge helpers
scored normalized positions, whose anisotropic scaling can change the raw
width/height-delta ratio. Physical evaluation explicitly inverts
reference_box_half_extent/reference_box_center and verifies true values against
raw data. Neither this coordinate-convention difference nor temperature-source
historical trajectory-estimator scores should be conflated. Do not choose the
estimator by observed R². Retain undefined predictions with valid/total counts.

Smokes must verify checkpoint hashes unchanged, required LJ relations, original
splits/data hashes, and coordinate parity with saved frame rows before broad
post-fit evaluation. Recover existing weights first; new training only for
irrecoverable missing artifacts. Evaluation scripts/artifacts are under
hpc/experiments/compact_lj_pratio/ and notebooks/results/compact_lj_pratio/.

### Frozen p-ratio evaluation launched after correctness checks — 2026-09-08

All70 existing checkpoints were found; no training reruns needed. Full post-fit
jobs4674041–4674110 submitted,1CPU16GB each,mendels_q,place=free:shared,no pins.
Exact seeds/studies/checkpoint hashes and immutable source snapshots:
compact_lj_pratio/jobs.jsonl,intended_matrix.json,execution_summary.json.
Evaluation runner SHA25665b2da9aa5e34df1d56b05a8e717696b11e3667974fb50275514485fc7daef7f.

Final smokes4674035(shared) and4674036(LJ-only spatial) completed400/100rows;
checkpoints unchanged; max saved coordinate-MSE differences3.64e-11/1.82e-11.
Independent raw-data p-ratio parity max5.32e-6 shared and2.67e-6 LJ-only.
Allfour raw dataset hashes verified. Actual-runner synthetic inverse test
includes a non-square reference and decoded graphs lacking normalization
metadata. Exact physical reference positions must define side groups: inverse
roundoff otherwise changes ties on regular LJ boundary grids. Inverse scales
come from untouched reference metadata, never the minimal decoded graph.

Preserved provisional/failed smoke attempts: mutable-path3989/3990;3995 wrapper
absolute-path prefix failure;4003 omitted mixed-T evaluation;4010/4011 metadata
loss in cloned graphs;4026/4027 wrongly required historical coordinate rows for
new frame150. Final evaluation checks all common historical frames25/50/100/199;
frame150 is additionally evaluated against raw physical p-ratio reference.
No failed provisional output enters the final70-run collection.

### Completed primary-metric backfill — 2026-09-08

All70 frozen AEs evaluated successfully; jobs4674041–4674110 exit0. All23,500/
23,500 network-frame predictions finite;235 source/frame cells with five seeds,
100/100 evaluations each (20 unique validation networks). Mixed-T remains
post-fit reconstruction transfer; no final test, propagator, or training rerun.
Exact source/seed tables, rawtrue/pred p-ratios, paired comparisons and provenance:
`notebooks/results/compact_lj_pratio/review.md` and its linked collection/ files.

Primary metric changes the interpretation: LJ reconstruction is meaningful but
moderate and horizon-dependent, not wholly absent. Standard shared2D gives
physical endpoint p-ratio R²0.451±0.020 at100,0.413±0.269 at199. MP2D4 gives
0.633±0.095 at100 but0.127±0.284 at199. Standard shared4D0.335±0.112/0.360±0.074;
spatial4D0.206±0.286/0.315±0.470; spatial8D0.437±0.317/0.406±0.254. Wider96ref/8D
late score0.457±0.195 is not an established winner. Increasing latent size is
not a monotonic solution. Spatial2D's slight coordinate improvement at199 comes
with paired p-ratio R² deterioration-0.453±0.248 (5/5 seeds worse). LJ-only D4
-0.105±0.323/-1.262±0.870 at100/199; LJ-only spatial worse still. Shared/LJ-only
normalization and exposure differ; do not interpret as isolated interference.

Other source responses remain strong despite coordinate residuals: shared2D
Reid0.931/0.956, low-T0.932/0.945, mixed-T0.840/0.898 at100/199. MP2D4 gives
Reid0.950/0.924, low-T0.988/0.991, mixed-T0.884/0.923. Exact SDs and100/100 counts
for every variant/source/horizon in review.md and source_frame_aggregate.csv.

All checkpoints unchanged; max savedcoordinate-MSE delta8.73e-11; max inverse
physical p-ratio reference discrepancy from rawdata5.32e-6. Physical primary and
historical model-coordinate metrics separately recorded. Do not equate these
endpoint scores with prior temperature-source trajectory estimators. No expert
inputs, losses, or checkpoint selection were introduced.

### Matched message-passing dimension and autonomous response study — 2026-09-08

User requested matched 2D message passing, autonomous propagation, and inline
source-wise p-ratio R² versus step plots comparing reconstruction with rollout.
LJ R² >= 0.4 at the declared nonzero horizons is the desired outcome, not a
checkpoint-selection rule or a promised result. Retain all negative/undefined
scores. AE reconstruction is an observed-frame reference, not a mathematical
upper bound on an observable score.

New AE: mp2_r16_d2, five seeds 3456456/123/456/786/2026, matching completed
mp2_r16_d4/code_v2 except latent dimension and output paths. Two encoder message
passing rounds, 16D static reference/node, hidden96, 32 decoder tokens. Full
200 frames (0–199), not just 100: 30 Reid + 30 low-T + 60 LJ trajectories,
24,000 frame samples/epoch, validation20/source, mixed-T20 post-fit only.
Required graph-distance-two/three LJ augmentation and exact existing splits
remain unchanged. Existing MP4D/8D checkpoints are reused. AE selection remains
worst-source coordinate validation reconstruction; no response callbacks.

Propagation design: frozen MP2 2D and 4D AEs, five seeds each, shared deltaMLP
hidden64 and mean static graph context projected16, with one-step delta loss
versus eight-step autonomous latent-state supervision (horizons1–8). Both learn
from the three retained sources only, full199 available transitions. State-only
validation loss selects propagator checkpoints; explicitly disable historical
default p-ratio selection. No expert inputs, physics losses, temperature/source
metadata, or final-test use. Reference initial state is frame0; no future
observations are supplied during autonomous rollout.

Report physical-coordinate endpoint directional-side p-ratio at25/50/100/150/199,
using exactly the same physical inverse, original reference side groups, and
20 validation networks/source as the audited AE backfill. Mixed-T remains
post-fit transfer. Save per-network true/predicted values, valid/total counts,
seed means/SDs, source-wise curves, checkpoint/code/data hashes and PBS recipes.
The completed MP4D reconstruction is 0.633±0.095 at100 but0.127±0.284 at199;
late representation error and propagation error must be distinguished.

Execution artifacts: compact_lj_mp2d/ and compact_lj_response_rollout/ under
notebooks/results; runners under matching hpc/experiments subdirectories.
Integration smokes gate full submission. Shared mendels_q, place=free:shared,
no host pins; full training8CPU16GB. Submission and completion status will be
recorded separately from this prospective recipe.

### Matched MP2D training submitted — 2026-09-08

Successful integration smoke4674631 exited0 in192.56s with frozen bundle
latent/decode reload parity and peak RSS5.39GiB. Smoke success preceded first
production submission; exact timing and identical training adapter hashes are
in compact_lj_mp2d/smoke_gate.json. Preserve failed adapter smoke/code_v1,
code_v2 successful smoke, and unsubmitted code_v3 bytecode-manifest attempt.
Production immutable code_v4 reuses compact reconstruction training source
manifest3fbd7c8d6e117ce462ef290fabef3592c2b786bb45a6b8325d4fb9f353eacfb9.

Five full AE jobs: seed3456456=4674650,123=4674651,456=4674652,
786=4674653,2026=4674654. Per-seed afterok p-ratio jobs4674655–4674659;
afterany collector4674660. Full runs observed around5.1–5.3GiB,8CPU16GB
shared mendels_q. Exact recipes, hashes, dependencies and attempts:
compact_lj_mp2d/jobs.jsonl and controlled_recipe.json. Scientific MP2D results
are pending. New inline reader: diagnostics/compact_lj_response_rollout.ipynb;
current completed curves are MP4D AE reconstruction only.

### Frozen MP2D/4D propagation matrix submitted — 2026-09-08

Final frozen harness code_rollout_v4 smokes4674692(one-step) and4674693
(eight-step) completed in175.08/173.31s with800 rows each:400 reconstruction
and400 autonomous evaluations, four sources ×20 validation networks ×five
horizons. AE state and all normalizers remained exactly unchanged. Independent
D4 reconstruction parity: max true p difference9.89e-17, predicted p7.61e-7,
coordinate MSE2.55e-11. Declared tolerances1e-5 p-ratio and1e-10 MSE reflect
float32/thread roundoff; failed earlier overly strict1e-10 p check is retained.
Frozen-source hash verification ignores runtime bytecode and verifies every
manifest-listed source file. No metric or model was selected using p-ratio.

Training source manifest3fbd7c8d6e117ce462ef290fabef3592c2b786bb45a6b8325d4fb9f353eacfb9;
runner SHA256d27b8464d5f4d4a7909223fbc27104d346f922fe1bfac24cdf678c9475d7423c.
Full20 jobs4674709–4674728: D2 jobs4674709–4674718 depend afterok on their
matching AE jobs4674650–4674654; D4 jobs4674719–4674728 use completed AEs.
Each dimension has seeds3456456/123/456/786/2026, one-step thenmultistep8
perseed in the job ordering. Width64, mean reference context16, batch32,
max30epochs/patience6, lr1e-4,wd1e-5, full199 transitions, equal-source loss.
Only state validation loss selects propagator checkpoints; frame0 is the only
observed dynamic state during autonomous evaluation. Required LJ relations and
original source splits retained. Mixed-T is post-fit only; final test untouched.

Live scheduler verification: ten D4 runs R, ten D2 runs dependency-held H.
Final collector/inline-notebook refresh4674730 is dependency-held afterany all
20 propagation runs, allfive independent D2 p-ratio evaluations, and initial
AE collector4674660. Frozen collector SHA256c3f5d95e964102c5529e0d61bd5120156d426f6f86a21fcb0b215852b4894225.
Exact identities/dependencies/resources in compact_lj_response_rollout/jobs.jsonl,
collection_job.json and submission_pbs_status.json. Scientific full results are
pending. Preserve all provisional smoke outputs and failed hash/parity attempts.

The results-reader notebook executed successfully with current MP4D AE curves;
missing MP2D and autonomous curves remain explicitly pending. It displays
source-wise seed mean/SD, valid/total and finite-R² seed counts with an LJ0.4
target line. Eight notebooks/68 code cells and eight data files passed the project
checker. Collector checks passed for exact cohorts, duplicate rejection, physical
target parity, perfect-prediction R²=1 and mean-prediction R²=0.

Submission provenance qualification: the consolidated rollout intended_matrix.json
was written after qsub and explicitly records that ordering; immutable runtime
code existed before submission, and each job writes its full resolved recipe
before fitting. Final smokes observed peak5,783,460kB;16GB requests retained.

### LJ data-budget diagnosis requested — 2026-09-08

User suggested more training to improve weak noisy-LJ reconstruction. Current
shared MP4D has60 LJ training trajectories,20 validation,120 reserved; all200
frames per training trajectory are already used. No reserve migration or new
training is part of this diagnostic. First evaluate physical endpoint p-ratio
reconstruction on all60 TRAIN LJ trajectories of the five frozen MP2R16D4
checkpoints, at25/50/100/150/199, and compare against their existing20-network
validation scores. Same exact estimator/inverse/original side groups as the
validated backfill; weights and normalizers fixed. Report target variance and
p-ratio MAE as well as R², since train and validation response spreads may differ.
This distinguishes evidence for a generalization gap from difficulty fitting
the response; coordinate training curves alone cannot answer that question.
Artifacts and exact provenance: notebooks/results/compact_lj_training_gap/.

Frozen LJ training-response diagnostic jobs were submitted before interruption: seed3456456=4674805.zeus-master, seed123=4674806.zeus-master, seed456=4674807.zeus-master, seed786=4674808.zeus-master, seed2026=4674809.zeus-master.
Evaluation-only: five frozen MP4D checkpoints, no new fitting or reserve access.
Runner SHA256 6497c5a6d17e88f2ca99a1e1955959e52b3fd99f235238245d6663a9b6250394.
Collection resumed after interruption; completed metrics are pending verification.

### Completed frozen LJ training-versus-validation response audit — 2026-09-08

Allfive evaluation jobs4674805–4674809 produced successful completion markers,
300 training rows each (60 training LJ networks ×five horizons). No fitting,
checkpoint selection or reserved-data access. Training checkpoints unchanged.
Historical PBS queries were intermittently unavailable; terminal success here
is supported by saved completed.json artifacts, not a claim of verified PBS
exit codes. Exact runner, data, split IDs, checkpoint, metric-module and collector
hashes are saved in compact_lj_training_gap/{jobs_submitted.json,collection.json}.
Collector SHA2563ae10e25a535353092af97cd8c6e919219ed8ece7e9753fa2a4ec6f694894709.

Physical endpoint p-ratio R² mean ±sample SD across seeds3456456/123/456/786/2026.
Every training cell has300/300 valid predictions (60 unique networks repeated
five times); every validation cell100/100 (20 unique networks repeated). Exact
train/validation IDs are disjoint, all five frames present, duplicates rejected.

| Frame | Training R² | Validation R² | Training p MAE | Validation p MAE | Training target variance | Validation target variance |
| --- | --- | --- | --- | --- | --- | --- |
|25|0.381±0.210|0.217±0.305|0.04922|0.04701|0.006743|0.004217|
|50|0.712±0.080|0.605±0.135|0.03988|0.03497|0.009735|0.004872|
|100|0.824±0.025|0.633±0.095|0.03755|0.03604|0.013043|0.005393|
|150|0.840±0.013|0.351±0.308|0.03825|0.04713|0.014937|0.005919|
|199|0.817±0.041|0.127±0.284|0.04222|0.05825|0.017069|0.007100|

Interpretation: late response reconstruction is possible on training networks;
there is a held-out response-error gap. Do not interpret the R² difference
alone as overfitting: validation target variance is smaller (about2.4x smaller
at199), and validation MAE is actually slightly lower at25–100. At150/199,
validation MAE also increases; late generalization is therefore a real concern,
not solely an R² denominator effect. These data motivate testing additional
independent LJ trajectories, but do not prove that more data will fix the gap.
Keep the validation cohort fixed and the120 reserved LJ trajectories untouched.
No additional training was launched by this diagnostic.

Existing MP4D histories selected coordinate checkpoints at epochs8–10 and
stopped20–22 under max60/patience12. Raising max_epochs alone would not bypass
that early stop, and coordinate histories do not establish p-ratio convergence.
The response-rollout notebook now includes inline training/validation R² and
MAE curves plus target-variance and count tables.

### Temporal linearity quick check — 2026-09-08

User asked when Reid/dePablo trajectories become linear. Definition: fit each
node coordinate against saved frame index, pool temporally centered SSE/SST;
20-frame sliding windows,20 existing validation trajectories per source. No
model training, response selection, or final-test analysis. All60 trajectories
and every window finite; exact IDs/code hashes/manifest in
notebooks/results/temporal_linearity_check/recipe.json, raw fits and summaries
alongside it. These are coordinate-linearity R², not p-ratio scores.

For all20 trajectories to pass in every subsequent window: threshold0.99 is
met from window start0 for Reid and low-T; stricter0.999 is met from Reid6
and low-T2. Mixed-T never meets either criterion through the last window180–199.
Whole0–199 straight-line fit median R²: Reid0.999939,low-T0.999833,mixed-T0.766289.
Mixed-T early/late20-frame window medians remain about0.18–0.23; no clear
noise-free linear regime. Minimum-image fractional-coordinate unwrapping using
each evolving box found zero crossings in these60 trajectories and preserved
results; this does not establish absence of wrapping elsewhere in the datasets.
Saved frame index is the reported unit; physical time per frame is undocumented.
Threshold-dependent empirical characterization, not a universal onset or proof
that the smaller motion components relevant to p-ratio are easy to predict.

### Engineering objective clarified — 2026-09-08

Engineering must use gradients through the frozen ML model to edit original networks, preferably with the AE alone and with the learned propagator if needed. LAMMPS is final verification only: no proposal scoring, acceptance, stopping, restart or winner selection. Final-design hashes precede verification. Protocol: `docs/research/engineering_goal.md`; audit: `network_engineering_log.md`. Reuse the earlier ML-only `engineering_endpoint_v1` baseline (3/3 Reid and1/3 low-T improved;6/6 valid;0/6 negative;one model seed), without duplicate runs. The negative-Reid53 animation is a historical simulator-guided design and does not meet this engineering objective. No new experiment launched by this clarification.

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

### Completed matched MP2D/4D autonomous response comparison

All five new MP2D reconstruction evaluations and twenty frozen-AE propagator
runs passed collection. Physical target cohorts, frozen AE reconstruction
predictions and checkpoint hashes matched the independent reconstruction
evaluations. Source-wise R² versus step, five-seed SD and valid/total counts:
`notebooks/results/compact_lj_response_rollout/review.md`; exact rows,
recipes, hashes, failures and job identities remain alongside the review.
No response-selected checkpoints or final-test data were used.


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
