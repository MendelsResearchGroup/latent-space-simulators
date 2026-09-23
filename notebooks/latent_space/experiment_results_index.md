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


### Shared MP2D response and raw-latent notebook — 2026-09-09

Created `notebooks/dynamics/04_shared_ae_response_and_latents.ipynb`: one shared reconstruction p-ratio R² plot for all four sources with five-seed mean/SD, raw z0–z1 trajectories, paths colored by physical frame response, initial latent versus final p-ratio, fixed-step latent/response correlations, strain associations and original-network descriptors. No new training or propagator results.

Raw-latent export uses predeclared seed786 of frozen `compact_lj_mp2d/mp2_r16_d2_s786_code_v4/ae.pt` (SHA25619fc84204be908dad3843ba38a1765bdecad7abf35d701aa50b3b23b0baa64f9), all200frames of20existingvalidationnetworks/source:80networks,16000/16000finite latentpairs. Mixed-T remains post-fit transfer. Physical endpoint p-ratio uses original physical frame-zero side groups, quantile.10; all400targets at25/50/100/150/199 exactlymatch independent saved evaluation. Per source3980/4000defined p-ratios:20frame-zero values remainNaN. No final-test evaluation, response-selection or new physics. Rawaxes are single-seed encoder outputs, notaligned/PCAtransformed acrossseeds.

Correlations are descriptive Pearson associations: initialz→finalphysicalp at199 uses20networks/source; framewisez→framep compares20networks at eachfixedhorizon, notpooledtimepoints. Strain heatmaps usefixedframe100; networkdescriptors useoriginalspringedges excluding LJaugmentedrelations. Undefinedconstantobservable correlations aremarked. These are not held-out predictionR², autonomouspositionpredictions or engineeringreliability evidence.

Allsevenfigures savedinline in eightexecutedcodecells, noerrors; nofigurefiles exported. ExactIDs, source/checkpoint/recipe/datahashes, code snapshot, CSVs, paritychecks and presentationhashes are in `notebooks/results/mp2_latent_analysis/`. Extractioncompletedlocally ononeCPU beforea proposedqueuehandoff; noPBSjob submitted. Plottingusescachedtables.

User requested excluding mixed-T from this notebook. Refreshed all seven inline plots and displayed tables to include only Reid, low-T dePablo and noisy LJ: 60 networks / 12,000 latent frames. The complete four-source export remains preserved; this is a presentation filter, with no new inference, fitting or evaluation. Current presentation hashes and the three-source plotting helper are saved alongside the export.

Added all-frame raw z0/z1 scatter plots against same-frame physical p-ratio and transverse strain, source-wise for Reid/low-T/LJ and colored by step. Per source: 3,980/4,000 p-ratio pairs (frame-zero undefined), 4,000/4,000 strain pairs. These are repeated trajectory observations; no pooled significance or predictive score is inferred. Nine inline figures saved; no new inference, fitting or physics.


### Shared MP2D early-step reconstruction evaluation — 2026-09-09

User requested the reconstruction p-ratio R² curve begin at step 1. Evaluate steps 1/5/10 from the same five frozen MP2D checkpoints (seeds 3456456/123/456/786/2026), with step25 as a per-network parity check against the existing physical evaluation. Preserve the exact20 validation IDs/source for Reid, low-T dePablo and noisy LJ; exclude mixed-T as requested. Use the established physical endpoint side estimator with original reference-side memberships (quantile0.10), saved normalizers and LJ graph-distance-two/three augmentation. No training, response selection, physics or final-test use. Selected-frame preprocessing avoids processing unrelated trajectories. Evaluation job4676379.zeus-master uses mendels_q, free:shared, 2CPUs/6GB. Results and provenance will be retained under `notebooks/results/mp2_latent_analysis/early_response/`; notebook04 combines these new early measurements with the existing25/50/100/150/199 curve.

Evaluation completed in job4676381.zeus-master:54s PBSwalltime,48s CPU,2452440KB measured memory (runner peakRSS2553172KB). All1200/1200 network/seed/frame pairs finite; each of60 source/seed/horizon summaries20/20valid,100/100 pooled per source/horizon. Step25 truth/prediction parity passed for all300pairs with maximum absolute difference9.8879238130678e-17. Dataset hashes match allfive trainingrecipes; exact IDs/checkpoint and recipe hashes, code snapshots and resource/job provenance retained in early_response.

New reconstruction R² mean±sampleSD acrossfive seeds at steps1/5/10: Reid−2.148572±3.648592 /0.064992±0.325833 /0.611693±0.073369; low-T dePablo−7.760463±2.936845 /0.613843±0.122322 /0.713498±0.061885; noisyLJ−11.030956±9.865822 /−3.709510±3.210145 /−1.853691±1.494054. Early negative scores are retained; no coordinate-error-only judgment or estimator change. Notebook04 now includes measured1/5/10/25/50/100/150/199 with five-seed SD. Its y-axis is linear in[−1,1], logarithmic outside, so both early negative scores and later positive response remain visible. Nine inline figures executed without errors.

Two launcher/output-directory failures preceded the collected evaluation and are retained in early_response/attempt_logs and metadata.failures:4676379 stopped at preflight before loading data;4676380 completed inference but failed at final directory creation before parity/CSV persistence.4676381 is the sole collected attempt. No model, metric, cohort or selection changes between attempts.


### Initial LJ line and network-level latent rates — 2026-09-09

User requested an explanation of the initial LJ latent line and trajectory slopes, with only supported associations shown in the notebook. Subsequent steering moves the reconstruction R² display start to step5 and separates plotting from evaluation: do not execute notebooks or inspect rendered images. Step1 evaluation artifacts remain preserved. Notebook04 now reads cached tables and contains the new compact plotting code directly; affected stale outputs were cleared. Syntax and table consistency checks passed, with no notebook execution.

Analyzed frozen representative seed786, exact20 existing validation IDs each for Reid/low-T/LJ (60 total, no mixed-T or final test). Reused all200-frame latent/physical exports; extracted original physical frame-zero geometry, spring stiffness, connectivity and orientation descriptors from the hash-matched datasets. No model inference, training, checkpoint selection or physics. A requested operational worker remained pending initialization, so root completed the extraction locally. The first extraction stopped because raw Reid lacked edge_multiplicity metadata; no rows were saved. Canonical unique-undirected edge normalization resolved it without changing physical positions. Source hashes, exact IDs, definitions, runtime/RSS and that failure are recorded under mp2_latent_analysis/structure_descriptors.

The initial LJ leading line accounts for94.3812% of its raw initial latent variance. Position along it correlates with mean initial physical y-position: Pearson r=-0.791107, Spearman=-0.625564, paired-bootstrap95% interval[-0.934178,-0.432567], leave-one-network-out range[-0.868139,-0.712960],20/20valid. This is a geometric placement association, not established mechanical response or proof of a translation artifact. A controlled intervention would be needed for a causal claim.

Trajectory rates are OLS dz/dstep with intercept over frames0–199. Against established final physical endpoint p-ratio at199: Reid dz0/dstep r=-0.969170 (bootstrap95%[-0.986493,-0.940581]); low-T dz0/dstep r=-0.993679 ([-0.997651,-0.988257]); LJ dz1/dstep r=-0.825622 ([-0.933511,-0.641386]). Each20/20valid. These descriptive correlations are not predictive R², reconstructed-position accuracy, autonomous rollout or engineering reliability. Geometric path slopes OLS z1 versus z0 correlate with finalp at about-0.307/+0.306/-0.625 respectively and do not survive the full search correction, so they are not highlighted.

Full audit contains612 source/latent/observable rows,600 evaluated and12 constant/incomplete. All four displayed findings pass a global maximum-absolute-correlation permutation correction across600 tests, using20000 independent within-source network permutations, seed20260909. Adjusted p: initialLJ0.011549; Reidrate0.000050; lowTrate0.000050; LJrate0.003200. All retain sign when any one network is removed. Bootstrap5000 paired-network resamples use the same seed; intervals are exploratory after selection, not independent confirmation. No p-ratio/strain was used to change model fitting or checkpoint selection. Full search, selected plotting rows, definitions, line directions, code/data hashes and code snapshots are under mp2_latent_analysis/correlation_search. The notebook replaces two broad scans with one four-panel view and preserves the earlier framewise/raw-latent views.

User requested local latent increments versus same-frame p-ratio. Added a cached-data-only notebook04 view of Δz/Δstep=(z_t−z_{t−1})/(t−(t−1)) against established physical p-ratio at ending frame t, steps5–199, no smoothing. Both raw latent coordinates shown for allthree retained sources in one2×3figure. Calculated1170 within-source/within-step correlations (195steps×3sources×2latents), all20/20valid network pairs. Median Pearson r for z0/z1: Reid−0.921508/−0.756935; low-T−0.977302/−0.792287; LJ−0.561741/−0.638375. Distinguish these from whole-trajectory OLS rates versus finalp; repeated frame pairs are not independent samples. Code, exact definition and hashes in mp2_latent_analysis/correlation_search/frame_rate_metadata.json and code/analyze_frame_rates.py. No new inference, fitting, physics or final-test use. Notebook syntax validated but not executed/rendered, per user instruction.


### Interior spring crossings in the Adam Reid showcase — 2026-09-09

User reported edges lying over one another inside engineered networks. Read-only audit of all10 saved original/optimized Reid design pairs for `lr0.08_b0.25_p50_mask_blocking_nodes` confirms proper interior crossings introduced by position optimization: original0/10 designs with crossings, optimized10/10. Crossing pair counts by ID:2=40,52=32,53=39,57=16,59=38,74=30,84=66,93=53,96=30,97=33. These are present in saved designs before LAMMPS; the audit excludes shared-node joints and wrapping edges, so it is independent of periodic rendering. It is conservative, omitting collinear overlaps, near contacts and periodic-edge intersections. No mixed-T or other-run validity claim follows.

The current geometry checks enforce length bounds, pairwise node separation and boundary memberships but omit spring–spring intersections. The reported physical p-ratios remain unchanged, but all10 audited optimized designs fail the newly explicit planar no-crossing requirement. Added that requirement to docs/research/engineering_goal.md and a concise failure annotation to engineering notebook01 without executing it. No optimizer implementation, frozen historical artifact, selection rule, simulation or notebook output was regenerated. A corrected constrained study must have a new frozen recipe and remain ML-only; geometric feasibility checks do not require LAMMPS.

Exact graph paths/hashes, crossing node-index pairs, counts, audit code and four passing analytic geometry checks are under notebooks/results/engineering_geometry_audit/. No images inspected or new jobs submitted.


### Planar joint-Adam rerun, prospectively declared — 2026-09-09

User authorized re-optimizing the same10 Reid and30 mixed-temperature networks with crossing prevention, followed by GIFs. Start from their original physical frame-zero networks, never the prior engineered graphs or matched low-T counterparts. Retain frozen model seed786 and historical calibrated z1 readouts; these scores are downstream response-calibrated proxies, not autonomous position predictions. Reid uses lr0.08/bound0.25/patience50; mixed-T uses arm-A lr0.02/bound0.25/patience50. Both: joint raw-gradient Adam, max1200steps, min_delta1e-5, radius4, unchanged topology/box/boundary nodes and strict lowest feasible frozen-ML score. Blocking-node masking is retained for Reid and explicitly added to the thermal arm-A hyperparameters to handle geometric obstructions. No physical-response-based stopping, editing, restarts or selection.

New PlanarConstraints checks all neighboring periodic images, proper intersections, nonincident contacts, and collinear overlaps while permitting ordinary shared-node joints. Accepted states require pair clearance min(0.05×median original spring length,0.5×original pair clearance), with a numerical tolerance floor. Analytic orientation roots detect contacts during the full linear interpolation from the previous accepted state, so an Adam move cannot jump through another spring and emerge crossing-free. Fixed periodic bond images and noncollapse are also enforced. Twelve geometry tests cover these cases, including a sampled independent orientation oracle. Full suite passed81tests and6subtests. A3-step Reid84 smoke preserved zero periodic contacts while reducing the frozen proxy0.04359483→−0.21449578; no LAMMPS was run for that smoke.

Freeze every final graph and all40 outcomes before final-only verification. Reid retains its120-increment athermal compression protocol; thermal retains the separately declared finite-T protocol,3paired Maxwell velocity seeds786/123/456 and200frames, with p-ratio from minus the transverse/axial strain-vs-frame OLS slope ratio. This is the existing declared thermal protocol, not a recovered original dataset generator. Every final physical trajectory will be audited for contacts; contact failures remain failures even if p-ratio is negative and cannot feed back into optimization. Save all40 comparison GIFs, including failures when trajectories exist, for4seconds in notebooks/engineering/Planar Adam engineering/. Display IDs are fixed before physical results: Reid93/84/96/97/52; thermal47/68/140/161/242/268. Notebook03 is a plotting/view layer over saved results; do not execute notebooks or inspect rendered images.

Exact cohorts, model migrations/tensor checks, frozen code/recipes, geometry audit, submissions, failures and results live under notebooks/results/planar_engineering/study_v1 and hpc/experiments/planar_engineering. Optimization uses shared mendels_q without host pins. Initial test import failed when MetaForge/src was missing from PYTHONPATH; adding the required auxetic dependency resolved it, and the full suite passed. No scientific recipe was selected by this environment repair.

### Dense Reid small-step study — prospectively declared, 2026-09-09

User supplied data/data_reid_with_angles and requested canonical assembly, integrity checks, AE displacement reconstruction R² by step, then100-step autonomous propagation. User capped the cohort at100networks. Planned70train/10validation/20finaltest with response-independent network-level splitseed123; preserve every available frame in the selected canonical dataset and use a contiguous0–100prefix for this study. Initial chunks contain1000frames rather than the requested/expected1500 and lackphysicaltime metadata; fullselected-cohortaudit pending, so no density-only-equivalence claim. Protocol: notebooks/results/dense_reid/protocol.md. Matched2D/4D × seeds123/456/786,Reference16MP2 AE,one-step/multistep8 deltaMLP; state-onlycheckpointselection. Postfit displacement-from-reference,consecutive-frame displacement,andprimaryp-ratioR² at every horizon1–100; validation andfinaltest keptseparate. Test only afterfitting allpredeclaredchoices; no test-driven retuning. Nojobs orresults areclaimed by this declaration.

Dense Reid preparation and launch completed. Selected100/100trajectories have1000frames each (100000/100000frames validated), not1500. Canonical data/reid_with_angles_1000_frames.pt SHA256f0ca7c612e3550fae8e52f58228d25f0bde369eb4ae04fc7cef07d7219d5dbcc; first101prefix SHA25657acf561f3e263fa4e87c52d161fbc87bd94ef7b785de0efbc7f9dc88fa7851b. Rawtensors retained,Box converted tocurrentgraph_utils class,provenance added; no inventedtime/temp. Fouredgecolumns validated as signedperiodicvector,length,stiffness; angularpotentialmetadata absent. Selected100 have0exacttopologyhashmatches to each historicalReidfile, so this is not established matched-frame-density resampling. All10000prefixtransitions change; medianabsolute coordinateincrement2.25544e-4 versusfloat32spacing2.38419e-7. OLSlinearpath residual/centeredSSEmedian1.19072e-4: earlynonlinearity exists but is small. Audit and sourcehashes: notebooks/results/dense_reid/audit/README.md anddata manifest. ConversionpeakRSS3.15GiB; standardnormalizedloader10100frames passed,peak939752KB,8.05s.

Smoke4677438.zeus-master completed bothpropagators, AEfreeze/normalizerassertions, andall101horizonmetrics on2train/2validation;60.41seconds,peak1484412KB,no finaltest. Frozen trainingharness/codehashmanifest2c126a8a25c071ca614899aa2bd435a35fbc7e9a3ee2f6de2f4cc65eab0c8f8e; preparedmanifest75b3d61adb3ef0c1f1b7b72fc583a99db23928217ba30c8ec54da834ff08feb4. Fullchains4677439/440/441:2D seeds123/456/786;4677442/443/444:4D same seeds. AllsixconfirmedR in mendels_q with8CPU16GBfree:shared. EachtrainsAE thenone-step+multistep8propagators andpostfitvalidation/testmetrics. Collector4677445 dependsafteranyallsix; exactjobs/hashes in dense_reid/jobs.jsonl,collection_job.json,launch_status.json. Notebook notebooks/dynamics/05_dense_reid_small_steps.ipynb reads saved tables,unexecuted. Userexplicitlyrequestedendingtheturnoncealltrainingisrunning; fulltrainingresults remainpending, no performanceclaim.

Planarengineeringupdate2026-09-09: userrequested dropping thelasttwo unfinishedoptimizations36/268 andproceeding. All10Reid+28thermal designsretained; cancelledcases remainexplicitin40-rowmanifest. Globalfreeze4677371completed,38finalverification/GIFjobsreleased; replacementcollector4677469queued. No physicalresponsefeedback, no result-basedcase selection. See network_engineering_log.md andstudy_v1/user_cancellation.json. Added unexecuted notebooks/engineering/03_planar_adam_engineering.ipynb forcachedresults andfixedGIFexamples. Physicalverification/GIFresults arepending.

### Dense Reid with angles: completed held-out evaluation — 2026-09-09

Terra confirmed allsixtrainingchains4677439–4677444 andcollector4677445 completedPBSexit0. All6AEs and12propagators have savedresults; nofailuretrace. Exact100networkresponse-independentcohort70train/10validation/20test; testnotusedfornormalization,fitting,checkpointselection. Each retainedframe0–100,2D/4DReference16MP2,modelseeds123/456/786,frozenAEandnormalizerassertionspassed. Bothone-step andmultistep8 trainedlatentloss; decodedphysicalmetricsarepostfit. Source=reid_dense. Data/split/codehashes andhistories remainindense_reid/; notebook05readscompletedcachedtables,notexecuted.

Atstep100 finaltest (seedmean±sampleSD; everymetric20/20networksperseed,60/60repeatednetworkevaluationsacross3seeds):

| Latent | Mode | Physical x/y displacement R² | Physical x/y increment R² | Physical endpoint p-ratio R² |
| --- | --- | --- | --- | --- |
| 2D | AE reconstruction | .918537±.000692 | .830400±.043647 | .933301±.022105 |
| 2D | Autonomous one-step | .918347±.001809 | .859778±.032175 | .821303±.040173 |
| 2D | Autonomous multistep8 | .918821±.001750 | .861021±.028288 | .816260±.045016 |
| 4D | AE reconstruction | .914588±.008256 | .828090±.026886 | .944904±.010001 |
| 4D | Autonomous one-step | .914155±.007564 | .867862±.049477 | .816709±.029053 |
| 4D | Autonomous multistep8 | .915601±.007155 | .871383±.039407 | .816356±.034519 |

Primaryresponseceilingandrolloutareseparate; no clearbenefitofmultistep8onstep100response. Earlystepsremainweak: step1meanphysicaldisplacementR²negativeforall6groups; bystep5means.613–.769. Retainfullearlycurvesandundefinedframe0scores. AE incrementsdifferenceconsecutivedecodedframes; autonomousfirstincrementstartsfromobservedphysicalframe0,sofirst-stepincrementerrorincludestheabsoluteinitialdecodingoffset. Subsequentrolloutincrementscompareconsecutivepredictions. MetricsareratioofsummedSSE/centeredcomponentSSToverpooledx/yphysicaldisplacements,noty-onlyR². Highlaterdisplacement/incrementR²doesnotaloneproveaccuraterecoveryofthesmallnonlinearresidual: dataauditlinearfitleavesmedian1.19e-4ofpositionvariance. All100framesandvalidcounts in per_seed_steps.csv/seed_summary.csv; conciseinterpretationinreview.md. No newtrainingorfinal-test-guidedretuningwasperformed.

### Shared propagator repair through100 steps — protocol,2026-09-09

User explicitly freezes the existing2D message-passing AE and prioritizes autonomous noisy-LJ p-ratio R² at100, approaching the AE reference. No AE training or dimensional sweep in this study. Existing five-seed validation at100: AE0.563±0.078, one-step rollout−1.887±0.559, multistep8−1.907±0.874; each20/20 valid perseed. Reuse exact existing training30Reid/30lowT/60LJ and20validation/source, graph-distance2/3 LJ augmentation, saved normalizers and three existing one-step initializations seeds123/456/786. Mixed-T and final test excluded.

Predeclared12-run matrix: full100-step autonomous latent-state loss versus decoded-position loss(+0.1latent), Adam lr1e-4/3e-4, three seeds,40epochs/patience8. Supervision1/2/3/5/10/25/50/75/100; evaluationevery1–100. Both select by equal-source decoded-position validation MSE, epoch0 eligible. No p-ratio/strain inputs, losses or checkpoint selection. Frozen decoder passes gradients only to propagator. Frame0 is sole observed dynamic input. Protocol and exact splits: notebooks/results/lj100_propagator/protocol.json (SHA25682cd088d7a3eef75520b9efa072d08fcfd2abc20a4905b29a4ba677d1f5a0d84).

Initial integration smokes4678412/4678413 completed, but review found source weighting, latent-stat warm-start preservation and dependency-freezing issues; those attempts are diagnostic only. Corrected immutable smoke required before production. Runner and PBS under hpc/experiments/lj100_propagator; all attempts preserved. Collector tools/analysis/collect_lj100_propagator.py retains all arms and failures, physical response R² and component-centered physical displacement R². Notebook notebooks/dynamics/06_shared_propagator_100_steps.ipynb reads cached tables; not executed. No production performance claim yet.

### Shared100-step propagator repair submitted —2026-09-09

Final immutable code_v6 smokes4678474/4678475 completed PBSexit0 in85.13/75.69runner seconds, peak5.737/5.729GiB. Each evaluates600rows (three validation networks, AE+autonomous, steps1–100),600/600finitepairs including2/3/5. Correctness comparison against18 historical rows (three sources,25/50/100,AE+rollout) measured maxtruep difference1.55523e-6 and maxpredictedp difference2.55157e-6. FrozenAE, normalizer and latent-stat digests unchanged. Puredecoded-position loss gradient topropagator sum9.685128; no latentloss included in this check. Initial smokes/code_v1–v5 remain preserved as diagnostic/failed or superseded attempts; v5 hardcoded parity field is not evidence and was replaced by actual v6 comparison.

Production12jobs4678481–4678492 all initiallyRunning, mendels_q/free:shared8CPU16GB24h, collector4678493Held afteranyall12,2CPU4GB. No AE fitting. Exact mappings and submission hashes in notebooks/results/lj100_propagator/jobs.jsonl; final smoke_gate.json records measured gate evidence. Recursive copiedLSS dependencies, runner/evaluator and protocol frozen under code_v6; manifestSHA25690b90dcfecbb817fd9ab68c0dfc3f655144c179c634c1e50b82bce05658c0432. Frozen collector and dependency ledger retained. Training uses100autonomous steps and nine supervised horizons; p-ratio evaluated only after selecting by decoded-position validationloss. Initialepoch0eligible; all four variants/three seeds retained.

Matchedthree-seed existing validationbaseline at100: noisyLJ AE0.594490±0.064363 versusone-step rollout−1.827949±0.729284; ReidAE0.956959±0.009619 versusrollout0.779318±0.038925; lowTAE0.991890±0.000178 versusrollout0.928330±0.051739. Each20/20perseed (60/60repeated,20unique/source). BaselineCSV saved without new inference. Productionresults pending; no claim of improvement from smoke. New notebook06_shared_propagator_100_steps reads futurecachedcollection; no notebook execution or figure export. User requested leaving full training oncluster after launch.

### Broader frozen-AE propagator study —2026-09-09,pre-submission protocol

User requested substantially different propagator approaches; keep frozen2DMPAE and focus100step physicalp-ratioR². Existing12 decoded/latentloss jobs4678481–4678492 remainrunning atsingleoperationalcheck, no tracederrors, collector4678493held. Do not duplicate orcancel them.

New36arms: residual-rateMLP, inferred-velocity/acceleration, GRUmemory, attentionoverlast8generatedlatents, directtimeMLP, cubiclatenttrajectory; eachmeanreferencecontext versuslearned4-queryattention, seeds123/456/786. Allmodelsfreshlyinitialized hidden64; reference tokens are16DfrozenH_ref+2referencecoordinates, nofutureobservations. Directtime/cubic are directtrajectoryprediction, reported separately fromrecursiveautonomous. Normalizedstepindex isknownclock, notobservedstrain. AE remainsunchanged. No dimensionchange inthispropagator-only study. Sameexact30Reid30lowT60LJtrain,20validation/source; nofinaltest/mixedT.

Commondecodedtarget-normalizedpositionMSE+.1normalizedlatentMSE at1/2/3/5/10/25/50/75/100,full100prediction; equal-source1/1/2minibatches, Adam3e-4wd1e-5,max60epochs/patience12, clip1. AllcheckpointselectionusespositionvalidationMSEalone,epoch0eligible,minimprove1e-5. Physicalp-ratioonlypostfit frompositions, originalsidequantile.10, every1–100, source-wise3seedmean/sampleSD andvalid/total. Reportallarms/failures; retainAEseparately.

CachefrozenAEembeddings/targettensorsonceperseed toavoid36repeatedrawdata/AEencodingpasses. Preparationvalidatescacheddecoderagainstcanonicaldecode onfirstvalnetwork/source at1/25/100 andfrozenAEhashes; no p/straincachefeatures. Code: src/lss/dynamics/trajectory_propagator.py andhpc/experiments/lj100_broad. ProtocolSHA25624284a0ab8b2dbca4f0ad6ec0e7020e853468e27679ee98b4772cc05df495dcc; exactmatrix/splits innotebooks/results/lj100_broad/protocol.json. Modeltests: all12family/context combinations preserveinitialstateexactly, nodepermutationinvariance, finite/nonzerogradients andcheckpointreloadparity (oneparameterizedunittest,0.190s); fullcachedintegration/per-familydecodergradient smoke requiredbeforeproduction. No broadperformanceclaims yet.

### Broad36-arm propagator matrix submitted —2026-09-09

Frozen code_v1 cache smoke4678635 completedexit0,77.66runnersec/86PBSsec,5.72GiB peak, cacheddecoderparity0.0. All12family/context smokes4678636–4678647 completedexit0,600rowseach,1.94–3.31runnersec,14–15PBSsec,0.70–0.97GiB peak. Root verifiedallframes1–100 andphysical pernetwork schema,AE/normalizer/latentstat invariants, exactcheckpointreload, purepositiongradientrange0.152276–0.475222; historicalAEp-ratioparityat25/50/100 maxabsolute1.99079e-6. Evidence root_smoke_review.json andsmoke_gate.json; no performanceclaimfromsmoke.

Productionaccepted: cache4678654(seed123),4678655(456),4678656(786) initiallyRunning8CPU16GB;36fits4678657–4678692 initiallyHeld afterokmatchingseedcache,4CPU4GB basedcachedsmokememory<1GiB; collector4678693Held afteranyall36,2CPU4GB. All mendels_q/free:shared, nohostpins. Fullcache retains120train/60validation, AE/data/norm provenance; previous12narrowjobsunchanged. RecursivecopiedLSS+runner/eval/prepare/collector/protocol immutablemanifestSHA2566478d2fd52eb2df5e0b6c270ab57b6e3977b894f300ea8a7a610f25b259cb3af. Exactarmmapping,dependencies,qsubcommands,submitterhash in lj100_broad/jobs.jsonl. Currentproduction submitter overrides frozenPBS8GB smoke request to4GB basedmeasurement; frozen scientificcodeunchanged.

Notebook notebooks/dynamics/07_broad_propagator_100_steps.ipynb reads cachedsource-wisemetrics aftercollection, displaysall6families withmean/attentioncontext andsourcewise100steptable. Directtime/cubic prediction remains distinguishedfromrecursiveautonomousrollout; frozen2DAEdoesnot implyGRUmemoryhas2Dstate. Notebooknotexecuted,nofigures exported. All36resultsarepending; noimprovementclaim. Endinteractivewaitingafterconfirmedsubmission aspreviouslyrequested.

### Three radical shared-dynamics hypotheses —2026-09-09,smoke submission

User explicitly requestedthree additionalradicalideas andqueueing, aimingoneLSS acrosssourceswithmixedTnevertrained. Newstudy lj100_radical keepssamefrozen2DMPAE,30Reid30lowT60LJtrain,20validation/source,three modelseeds123/456/786 andsharedlearnedgraphconditioning; reusescompleted lj100_broad featurecaches ratherthanencodingagain.

Predeclared9models: (1) graph-conditioned stable8Dliftedlineargenerator, A=.05(B−Bᵀ)−.1diag(softplus(diagB)+.001), exactmatrixexponential andlearnedlinearreadoutto2Dlatent; (2) soft4expert graph/time trajectorymixture withgraph-derivedweights,no datasetIDs; (3) conditionalflowmatching over200Dfuturelatentdisplacementpath,graph+initiallatentcondition,8fixedantitheticnoisesamples(seed91823),16Eulerflowsteps,meanofdecodedpositions forpointprediction, notbestof8. Flowmotionscale istraining-onlycomponentRMS,floor1e-3. Allinternalstate/lift/distributiondimensions areexplicit, notclaimingentiredynamicsstate2D. Allreferencecontextuses4queryattentionoverfrozen16DH_ref+2referencecoordinates.

Operator/experts train decodedpositionMSE+.1latentMSE; flow trainsdecodedpositionMSEofestimatedflowendpoint+.1flowvelocityMSE. SameSUP1/2/3/5/10/25/50/75/100, Adam3e-4wd1e-5,sourcebalanced1/1/2,max60epochs/patience12,minimprove1e-5. Selectbythree-sourcepositionvalidationMSEalone,epoch0eligible; flowusesfixed8sampledecodedmean forvalidation. p-ratio/strain/sourceID/temperature excludedfromlearnedinputs/loss/checkpointselection. Operator/expertspredictionandflowensemblemean labeledseparately.

MixedTpostfitonly: exact20existingvalidationIDs fromAErecipe mixed_evaluation; nofinaltest. Freezeallpredeclaredfitcheckpointsbeforemixedcache/evaluationchain. Reportphysicalendpointp-ratioR² andseparatelyprefix0..t negative ratiooftwoOLSstrain-vs-frame slopes,originalsidequantile.10,atall1–100; do notchoosefavorableestimator. Flowpositionensemblespreadsaved; noresponse-basedsampleselection. Cacheprovenancefrombroad preserved; newprotocolSHA256ecd0afbde12b6843e4b9d0367e3a28ad96c06b3bf74e061d22e86e56024d3ae5.

Canonicalmodels src/lss/dynamics/radical_propagator.py; harnesshpc/experiments/lj100_radical. Smalllocalmodelchecks passed3families: exactinitialstate,nodepermutationinvariance,finite/nonzerogradients,flowtrainshape/finiteobjective. Frozen code_v1 smokes:fits4678753(operator),4678754(experts),4678755(flow),mixedprep4678756afteranyall3,mixedevals4678757–4678759afterokprep. Production9fits+postfittransfer awaitintegrationreview; noresultclaimfromsmoke. Existing12+36runsleftunchanged.

### Completed48-run sharedpropagator evidence and radicalbudget revision —2026-09-09

Both lj100_propagator(12/12) andlj100_broad(36/36) collectors reportcomplete. Rootreadcachedseed_summary.csv only, no additionalinference. At100, warm-startdecodedpositionfine-tuning lr3e-4: LJ−0.329599±0.132810, Reid0.835348±0.018367,lowT0.933744±0.032438; each60/60valid across3seeds,20unique/source. MatchedoriginalLJrollout−1.827949±0.729284; improvementdoesnotmeet0.4goal. Allotherarms retainedinrespective review.md/seed_summary.csv. HighestreportedbroadLJmean ishistory_attention/mean0.014119±0.175893, withReid0.541744±0.574589 andlowT0.143345±0.236871; noall-sourceworkingmodelclaim. This is retrospectivevalidationdescription, notresponsecheckpointselection orfinal-testclaim.

Smallcachedcasecheck(seed123 only): direct_mlp/mean predictedp-ratioSD at100 is0.029206 vsGT0.075345 forLJ; Reid0.028520vs0.203294,lowT0.065369vs0.371283. history_attention/mean andgru/attention alsohavecompressedpredictedspread; checkedbestepochs58and59,trained60. This supportsa hypothesisofunderlearnednetworkvariation andpossiblebudgetlimitation, notproofthatallnetworks/modelsconvergedorarchitecturealoneisresponsible. All36freshmodels versus12warm-starts alsohaveinitialization/budgetdifferences; don'tclaimmatchedcausalarchitecturecomparison.

Radicalv1all7smokejobs4678753–4678759 exited0; source600rows/model,mixed4000/model; AE/statinvariantsandcheckpointreloadpass. Purepositiongradientoperator.00711/experts.15826/flow7.61260. Mixedprep55s/3.48GiB,transfer22–55s/~.76GiB. Beforeanyradicalproduction, increaseallthree newfamilies maxepochs240/patience30 basedabovebudgetevidence, leavingobjectiveandselectionunchanged; preservev1smokes. Newimmutablecode_v2smoke gatesproduction. Finalnewprotocolhash below; this is notmatched60epochcomparison. No mixedTresultenteredloss/modelselection orbudgetchoice.

Radical revised protocolSHA256 10365d3e13ee35f9a251674fdf73212f7ea331fe6580f4f9d8f98c7913cee884.

### Radical9-run shared-model study submitted —2026-09-09

Finalimmutablecode_v2 smokesall7passed (source600rows/model,mixed4000/model),includingpuredecodedpositiongradients,frozenAE/normalizers/stats,andexactreload. Scientificmodel/evaluatorunchangedfromv1; budgetreadfromfrozenprotocol240epochs/patience30. Fullprovenance smoke_gate.json andjobs.jsonl; code_v2manifestSHA25604007ea9a3113862b48f7ef3eb1a781f2b3089c8fe0ef5ca7a6dec6b184c1a9a.

Production9fits4678881–4678889 initiallyRunning: threefamilies×seeds123/456/786. Mixedpreparations4678890,4678894,4678898 Held afteranyALL9fits, ensuring post-fit-only mixedTaccess. Transferjobs4678891/92/93,4678895/96/97,4678899/4678900/4678901 Held afterokseedmixedcache; eachrequiresits savedcompletedmodel/hash beforeevaluating. Collector4678902 Held afteranyall9transferjobs; all22jobsaccepted, exactdependenciesverifiedagainstledger. Sharedmendels_q/free:shared, nohostpins; fit4CPU8GB,mixedcache8CPU16GB,transfereval4CPU4GB,collector2CPU4GB. No finaltestuse, no p-ratio/strainselection, no mixedTtraining. Existingbroadcache reusedwithits source/AE/datahashes. Rootdidnotwaitproductionresults; remainpending.

### Radical study interim step100 collection —2026-09-09

User requested completionstatus. Eightofnine productionfit completed artifacts present: all3graph_operator,all3soft_experts,flowseeds123/456; flow786notyetcompleted atthisread. Mixedtransfer/finalcollector notyetcollected. Rootreadonlyexistingstep100CSVcolumns,nofitting/inference. Progressfiles lj100_radical/progress_step100_per_seed.csv andprogress_step100_summary.csv explicitlyretaincompleted-seedcounts.

Graphoperator3seedmean±SD p-ratioR² at100: Reid0.853944±0.070522,lowT0.954230±0.007798,LJ−0.301094±0.147422. Softexperts: Reid0.868665±0.012460,lowT0.940654±0.008642,LJ−0.133820±0.011512. Eachsource60/60valid,20uniquenetworks/seed. PairedAEreconstructionReid0.956959±0.009619,lowT0.991890±0.000178,LJ0.594490±0.064363. Two-seedflowONLY: Reid0.118191±0.151114,lowT0.078314±0.170099,LJ−0.332200±0.062188,40/40valid/source; thirdseedpendingso provisional. Allmetricsvalidation,physicalendpointsidesquantile.10, nofinaltest. NoLJsuccessclaim; mixedTpostfitremainsunavailableatthischeck. Longerbudget/operator/experts preserveother-source responsebetter thanearlierfresh60epochmodels, butbudget/architectureconfounded; do notattributecausally.

### Completed radical LJ comparison —2026-09-09

User redirected focus exclusivelytoLJ, leavingexistingmixedjobsalone. All9radicalfitscompleted; rootcollectedsavedLJrowsat25/50/100, no newinference/fitting. Fullthree-seed100stepphysicalendpointp-ratioR²: graphoperator−0.301094±0.147422; softexperts−0.133820±0.011512; flowdecodedensemblemean−0.318065±0.050330. PairedfrozenAE0.594490±0.064363. All60/60valid perapproach/horizon (20validationnetworksperseed,20unique); nofinaltest. At50,operator−0.328515±0.219204,experts−0.374112±0.230606,flow−0.249618±0.061552,AE0.443255±0.170377. None meetsLJ0.4goal. All3seedflow replacesprevious2seedinterim, withoutdiscardinganyseed. Exactrows/summary lj100_radical/lj_completed_per_seed.csv andlj_completed_summary.csv. BroaderfreshstudybestLJ100mean0.014119±0.175893 likewisebelowtarget. Architecture/training-budget differencespreventcausalranking. Nextpriorityisdiagnosingmissingnetwork-specificresponse ratherthanclaimingAEfailure frompropagationerrors.

### Real three-frame observed-history experiment —2026-09-09

User explicitly authorizesfirst3observedframes, testingwhethermissinginitialmotion explainsLJpredictiongap. PreviousbroadGRU/historyattentionusedgeneratedhistoryonly; thisnewinputcontractallowsactualframes0/1/2, thennoobservedfuture. Update docs/research/dynamics_goal.md toseparatelypermitthisdiagnostic; do notclaimitmeetsinitial-frame-onlygoal. Originalreference, AEencoder/decoderandnormalizers stayfrozen; sourcepipeline/LJhop2/3 unchanged.

Matched12fits: history_mlp versusvelocity_residual ×observedframecounts1or3 ×seeds123/456/786. Botharchitecturesconditiononlearnedattentionreferencecontext,z0,lastobservedz,andtwoinitiallatentincrements(normalizedbytraining-onlyRMS,floor1e-5). One-framecontrolgetsonlyz0,zeroincrementfeatures; samemodelwidth/parametercount. velocity_residual addsexplicitconstantobservedvelocity (z2−z0)/2when3framesavailable,otherwisezero; learnedcorrectiononallmodels. Alsoevaluateuntrainedlatentvelocityextrapolationfromthreeframes. Prefixencodedrelativetoframe0, positions/graphonly, nop/strain features.

Useexistingbroadcached120train/60validation exactly (30Reid30lowT60LJtrain20val/source),nofinaltest/mixedT. Equal-source1/1/2batch,Adam3e-4wd1e-5,max240/patience30,minimprove1e-5,positionMSE+.1latentMSE; state-onlycheckpointselectionatforecastSUP3/5/10/25/50/75/100,epoch0eligible. Scoreonlyforecastframes3–100forbotharms,neverobserved0/1/2. Primaryabsolute100means100stepsfrom0or98afterobserved2; reportconditioninglengthandforecasthorizonexplicitly. Physicalendpointsidep-ratioR²andAEreconstruction,linearhistorybaseline,mean±SD/validtotalallretained. Hypothesisratherthanclaimofhistorysuccess.

ProtocolSHA2568eec0f69a96bb8c85bdb8ab8136fe85c9e48b5fa641159a7d6a12cf4873b3526; canonicalmodel src/lss/dynamics/observed_history_propagator.py, harness hpc/experiments/lj100_observed_history. Two smalltests/fourmodelsubcasespassed prefixpreservation,future-dataexclusion,nodepermutation,finite/nonzerogradients,observedmotionsensitivity,andanalyticlinearpathbaseline (0.046s). Smokes4679050(historyMLPh1),4679051(historyMLPh3),4679052(velocityresidualh1),4679053(velocityresidualh3),4CPU4GBsharedmendels_q. Expected588rows/h1and882/h3 for3smokevalidationnetworks×98forecastframes×2or3kinds; decodedgradient/frozen/reload/scoreboundarygate beforeproduction.

### Observed-history12-run matrix submitted —2026-09-09

All4frozen code_v1 smokes4679050–4679053passed: h1rows588,h3rows882,exactforecastframes3–100,noconditioningframescounted. Kindsh1AE/prefixprediction;h3also linearhistorybaseline. Puredecodedpositiongradienthistoryh1.256259,h3.375347;velocityh1.256259,h3 86.3798. FrozenAE/normalizers/latentstats unchanged; exactcheckpointreload. One-framehistory/velocity modelsaremathematicallyidenticalwhenobservedvelocityiszero; retainedpairedcontrolsarenotindependentarchitectureevidence.

Production12fits4679056–4679067 initiallyRunning4CPU4GB/mendels_q/free:shared; collector4679068Held afteranyall12,2CPU4GB. Exactarmmapping/resources/deps in jobs.jsonl/production_status.json under lj100_observed_history. FrozenrecursiveLSS/harnessmanifestSHA2569ecb71c7842212de6768b2fbac8dd157b926ac88a9b0703bbc79d56807a8721d; current submitterSHA2564c80ef9ffbe0bcb3fa65ebc6b65b2c2a2fa4c35487d035ec68ae85b0266a287f. Allcache/data/source/cohort hashespreserved in recipes; no AEfit/newdata/no mixedT/no finaltest. Notebook08_observed_history_100_steps.ipynb reads futurecachedcollection,notexecuted. No improvementclaim; productionpending,endinteractivewaitingafterlaunch.

### Observed-history overnight results completed —2026-09-10

Cachedcollectionconfirms12/12completed, protocolSHA2568eec0f69a96bb8c85bdb8ab8136fe85c9e48b5fa641159a7d6a12cf4873b3526. Rootreadseed_summary.csvonly,no newinference/jobs. Atabsolute100 (98predictedstepsafterobservedframe2), LJphysicalendpointp-ratioR²3seedmean±sampleSD: historyMLPoneframe−0.082002±0.104242,threeframes−0.000415±0.260152; velocityresidualthreeframes−0.204202±0.147521; untrainedthreeframelinearhistory−26.823520±24.097677; pairedfrozenAE0.594490±0.064363. Everyentry60/60valid over20uniquenetworks/3modelseeds. One-framehistory/velocityarmsareidenticalcontrols, notindependentarchitectureevidence. NoLJtargetsuccessorrobusthistorygainclaim; thisdoesnotruleoutotherobservationwindows orhistoryrepresentations.

HistoryMLPReidoneframe0.879374±0.019508→threeframes0.914041±0.012291; lowT0.942840±0.009559→0.963267±0.026731. ThreeframevelocityresidualReid0.885450±0.004009,lowT0.954053±0.022110. Validationonly,20/20/source/seed; nofinaltest/mixedT. Scoredboundary3..100 excludesobserved0..2. Fullall-armcurves/counts in lj100_observed_history/per_seed_steps.csv,seed_summary.csv,review.md; notebook08readscachedoutputs,notexecuted. No newexperimentlaunched bystatusrequest.

### Observed-velocity recurrent acceleration pilot — 2026-09-10

User requested the previously untested combination of three observed initial frames and recurrent acceleration. One seed123, shared frozen MP2D AE, existing broad cache and unchanged exact30Reid/30lowT/60LJ training plus20 validation/source; no mixed-T or final test. Only actual frames0/1/2 are supplied. Initialize z at frame2 and per-frame velocity v=(z2-z0)/2; predict a_t from reference attention64, current generated z, scaled generated velocity, z0 and known step; v_next=v+a, z_next=z+v_next (dt=1), through absolute100. Acceleration output scaled by training-only prefix increment RMS/100, zero-initialized final layer starts at constant-velocity baseline. No observed future or expert observable supervision.

Reuse observed-history runner/evaluator: decoded-position MSE +0.1 latent MSE at3/5/10/25/50/75/100; Adam3e-4, wd1e-5, clip1, max240 epochs/patience30; checkpoint equal-source position validation MSE only, epoch0 eligible. Post-fit physical endpoint side p-ratio R² for all frames3–100, original side quantile0.10 and frozen physical inversion; report AE reconstruction, generated forecast and linear-history baseline separately, source-wise valid/total. Single-seed preliminary results, no seed SD; compare matching seed123 historical results. No notebook.

Protocol SHA256 2815cc80352e3db6f6fd8a368cfc626e852782ec08780a207d748e7394b9e77d. Harness hpc/experiments/lj100_observed_acceleration; canonical model src/lss/dynamics/observed_acceleration_propagator.py. Smoke4680412.zeus-master submitted; one production fit plus collector to follow only after smoke gate. Source/data/checkpoint/cache hashes and exact cohorts inherited and recorded in recipe; frozen snapshot and PBS identities recorded in jobs.jsonl. Results pending.

Root targeted recurrence check passed: observed prefix exactly preserved; zero acceleration matches constant observed velocity; nonzero constant acceleration matches analytic semi-implicit Euler z2+k*v0+k(k+1)/2*a; node permutation invariance and finite nonzero acceleration gradients. No full notebook execution or production inference locally.

Smoke4680412 exited0:882 rows across3 smoke validation networks ×98 forecast frames ×3 kinds; pure decoded-position gradient23.72935, frozen AE/normalizers/stats identical, checkpoint reload maxabs0. Runner2.5753s, PBS11s, peak resident821884KB. Frozen code_v1 manifest SHA2568dd9059f5009a9fa4a60de0dd558c2370864f3b5c0c107bb46f657eed44d7197. Production fit4680417.zeus-master accepted Running (4CPU4GB); collector4680418.zeus-master accepted Held afterany fit (2CPU4GB). Both mendels_q/free:shared, no host pins. Production results pending; no training wait or performance claim.

### Local dynamics versus rollout drift diagnostic — 2026-09-10

User requested true-input one-step versus free-rollout tests, including smaller/larger propagators and velocity/graph-conditioning alternatives. Six one-seed123 arms: state32/state64/state128 (only current latent, time and graph); velocity64/velocity128 (also rolling three-frame velocity/history); velocity_node64 (current-state-conditioned attention directly over reference node tokens rather than fixed pooled context). Width varies only the two-hidden-layer increment MLP; standard reference context remains64D. Node-attention arm changes pooling/parameterization together, so it is a graph-conditioning alternative, not a perfectly parameter-matched causal test. Model parameter counts saved in recipes.

AE frozen, exact existing broad seed123120train/60validation cache reused;30Reid/30lowT/60LJtrain,20val/source; no mixed-T/final test. Training samples8 true consecutive prediction targets per trajectory each epoch across3..100; windows end before target. State-only arms explicitly mask history; velocity=(z_t-z_t-2)/2. Predict next latent as z_t + training-increment-RMS * MLP; final layer zero starts at persistence. Loss decoded-position MSE +0.1 latent next-state error standardized by training-only RMS increments over1..100 (equivalently teacher increment error). Shared source-balanced1/1/2; Adam3e-4wd1e-5clip1,max120patience20. Checkpoint equal-source teacher one-step decoded-position validationMSE at3/5/10/25/50/75/100,epoch0eligible,minimprovement1e-8; no observable selection.

Each frozen model gets (a) teacher_one_step from true rolling windows, (b) prefix_prediction recursively from only actual0/1/2, (c) AE reconstruction, (d) persistence_one_step copying actual previous latent, (e) velocity_one_step extrapolating actual recent-window velocity one step. These five evaluator kinds are authoritative; inherited generic baseline prose in protocol is specialized here to the rolling true-input one-step baseline, not a second long extrapolation. All positions evaluated with original physical inversion/endpoint directional sides quantile0.10 at3..100. Record p-ratio R², displacement R², valid/total and auxiliary latent-increment R² perframe and aggregated; one-step skill must exceed persistence rather than just match AE on near-identical frames. Velocity/node/width effects preliminary single-seed, no seed SD. Teacher inputs are diagnostic access to true history and are never presented as free forecasting.

Protocol SHA256 fdba39bdc2847b5e327ce16ea4383be4524b38cff0c7820c3267058c76647d62. Canonical src/lss/dynamics/local_dynamics_propagator.py; harness hpc/experiments/lj100_local_dynamics; no notebook. Smokes4680467–4680472 submitted; production waits all-six integration gates. Existing acceleration job untouched.

All six smoke jobs4680467–4680472 exited0:1470 rows each (3networks×98frames×5kinds), frozen AE/normalizer/stats invariant, exact reload0, pure position gradients0.00151662/0.00209710/0.00277609/0.00208393/0.00282843/0.00204045 in matrix order. Runner3.72–4.26s,PBS13–14s,peak RSS827460–901660KB. Root test tests/test_local_dynamics_propagator.py passed all six model subcases in0.668s, covering no future leakage, masked-history controls, permutation invariance, prefix/persistence identity and finite gradients. Frozen code_v1 manifest SHA2569626d3eab486c4d741724678b4378e993d63b60c8aea0cfac438a647be6b4d76. Production state32/64/128 jobs4680482/4680483/4680484; velocity64/128/node64 jobs4680485/4680486/4680487 all accepted Running4CPU4GB. Collector4680488 Held afterany all six,2CPU4GB. All mendels_q/free:shared,no host pins. Stop after submission; results pending, no performance claim.

### Local diagnostic completed — 2026-09-10

Status reconciliation: all six fits4680482–4680487 and collector4680488 PBS finished exit0 with completed artifacts; collection complete6/6. Root recipe check: all six trained20epochs and selected epoch0 (zero final layer = persistence); runner181.57–259.74s. Thus identical outputs across architectures reflect checkpoint selection of the untrained persistence baseline, not evidence of equivalent learned dynamics or successful one-step learning. Step100 source-wise p-ratioR² identical all six (each20/20valid,seed123): AE Reid0.9515/lowT0.9919/LJ0.5202; teacher_one_step and persistence_one_step Reid0.9514/lowT0.9917/LJ0.5185; velocity_one_step Reid0.9515/lowT0.9919/LJ0.5205; free prefix_prediction Reid−0.6718/lowT0.3978/LJ−0.6904. Values rounded4decimals here; full precision per_seed_steps.csv. No claim that learned one-step prediction works: it did not beat persistence under the declared position-validation selection. Frozen checkpoints and failed-to-improve training histories retained. Existing acceleration4680417 still Running and collector4680418 Held at this check; no new jobs or inference.

### Pure latent-increment diagnostic rerun — 2026-09-10

User requested dz loss for the recent six models. New lj100_local_dz preserves six architectures state32/state64/state128/velocity64/velocity128/velocity_node64, single seed123, exact cached splits and frozen AE/normalizers, 8 sampled true consecutive targets per trajectory/epoch, Adam3e-4wd1e-5clip1,max120patience20. Only objective and checkpoint criterion change: pure componentwise training-RMS-scaled dz MSE, validation same dz MSE at3/5/10/25/50/75/100 averaged equally over sources; epoch0 eligible, minimum improvement1e-8. Under teacher input dz error equals next-state error, so runner uses algebraically equivalent prediction-minus-target expression. No decoder in training/selection; frozen decoder only in post-fit physical evaluation. No p-ratio/strain supervision or selection.

Compare teacher one-step and free98-step rollout after actual0..2, plus frozen AE and teacher persistence/velocity baselines, all3sources20val/source, p-ratioR² valid/total and latent-increment R². No mixedT/finaltest; one seed preliminary, no seed SD. Previous study selected epoch0 all six; this rerun changes both objective and selection, so cannot isolate which caused any difference. Protocol SHA256 f5fccdd9ad13cf67b69ad967bc58e3cab188d06b2c5005156b397c5efe1dc792. Harness hpc/experiments/lj100_local_dz reuses canonical local_dynamics_propagator unchanged. Six integration smokes gate six production runs+collector; no notebook or production wait.

Small cached-history review of prior lj100_local_dynamics (no inference): equal-source val_latent mean initially2.07658 in every arm; minimum within saved20epochs at epoch20 was state32 .40563,state64 .35681,state128 .31367,velocity64 .09495,velocity128 .03642,velocity_node64 .09792. Thus latent-increment validation improved substantially while position selection retained epoch0. Histories establish local loss improvement but only selected epoch0 weights were saved, so no post-fit p-ratio/free-rollout claim can be made for those discarded trained states. New pure-dz rerun tests retained increment-selected checkpoints.

Pure-dz smoke4680538–4680543 all exited0,1470rows each,pure latent gradients9.7148/12.9502/22.6006/14.7995/31.1128/17.3819 (matrix order),frozen invariants and exactreload0. Runner3.71–5.28s,PBS13–14s,RSS746256–750352KB. code_v1 manifest SHA256440500a95981dc0b7ba602bf80f4b7e93e39c76e6ab364338c6e3542b696ce1a. Production state32/state64/state1284680548/4680549/4680550,velocity64/velocity128/velocity_node644680551/4680552/4680553 accepted Running4CPU4GB; collector4680554 Held afterany all6,2CPU4GB; shared mendels_q/free:shared. Results pending; stop after confirmed launch.

### Pure dz and observed acceleration overnight results — 2026-09-11

Completed artifact collections: lj100_local_dz 6/6 and lj100_observed_acceleration 1/1; no failed artifacts. Cached metrics only, no new inference/jobs. Single seed123, validation20/20 per source per reported row; no seed SD, mixedT or finaltest. Absolute100 free rollout has98 predicted steps after actual0..2. Both protocols/snapshots/jobs retained above.

| Study | Family | Mode | Source | Step100 p-ratio R2 | Valid/total |
| --- | --- | --- | --- | --- | --- |
| lj100_local_dz | state32 | prefix_prediction | depablo_low_temp | 0.667568 | 20/20 |
| lj100_local_dz | state32 | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_local_dz | state32 | teacher_one_step | depablo_low_temp | 0.991719 | 20/20 |
| lj100_local_dz | state32 | prefix_prediction | lj_noisy | -1.987189 | 20/20 |
| lj100_local_dz | state32 | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_local_dz | state32 | teacher_one_step | lj_noisy | 0.519874 | 20/20 |
| lj100_local_dz | state32 | prefix_prediction | reid | 0.213811 | 20/20 |
| lj100_local_dz | state32 | reconstruction | reid | 0.951530 | 20/20 |
| lj100_local_dz | state32 | teacher_one_step | reid | 0.951251 | 20/20 |
| lj100_local_dz | state64 | prefix_prediction | depablo_low_temp | 0.853394 | 20/20 |
| lj100_local_dz | state64 | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_local_dz | state64 | teacher_one_step | depablo_low_temp | 0.991700 | 20/20 |
| lj100_local_dz | state64 | prefix_prediction | lj_noisy | -2.778996 | 20/20 |
| lj100_local_dz | state64 | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_local_dz | state64 | teacher_one_step | lj_noisy | 0.519131 | 20/20 |
| lj100_local_dz | state64 | prefix_prediction | reid | 0.350394 | 20/20 |
| lj100_local_dz | state64 | reconstruction | reid | 0.951530 | 20/20 |
| lj100_local_dz | state64 | teacher_one_step | reid | 0.951810 | 20/20 |
| lj100_local_dz | state128 | prefix_prediction | depablo_low_temp | 0.838624 | 20/20 |
| lj100_local_dz | state128 | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_local_dz | state128 | teacher_one_step | depablo_low_temp | 0.991818 | 20/20 |
| lj100_local_dz | state128 | prefix_prediction | lj_noisy | -5.117549 | 20/20 |
| lj100_local_dz | state128 | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_local_dz | state128 | teacher_one_step | lj_noisy | 0.518847 | 20/20 |
| lj100_local_dz | state128 | prefix_prediction | reid | 0.395671 | 20/20 |
| lj100_local_dz | state128 | reconstruction | reid | 0.951530 | 20/20 |
| lj100_local_dz | state128 | teacher_one_step | reid | 0.951918 | 20/20 |
| lj100_local_dz | velocity64 | prefix_prediction | depablo_low_temp | 0.343384 | 20/20 |
| lj100_local_dz | velocity64 | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_local_dz | velocity64 | teacher_one_step | depablo_low_temp | 0.991849 | 20/20 |
| lj100_local_dz | velocity64 | prefix_prediction | lj_noisy | -5.641306 | 20/20 |
| lj100_local_dz | velocity64 | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_local_dz | velocity64 | teacher_one_step | lj_noisy | 0.520271 | 20/20 |
| lj100_local_dz | velocity64 | prefix_prediction | reid | -0.328182 | 20/20 |
| lj100_local_dz | velocity64 | reconstruction | reid | 0.951530 | 20/20 |
| lj100_local_dz | velocity64 | teacher_one_step | reid | 0.951880 | 20/20 |
| lj100_local_dz | velocity128 | prefix_prediction | depablo_low_temp | 0.564907 | 20/20 |
| lj100_local_dz | velocity128 | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_local_dz | velocity128 | teacher_one_step | depablo_low_temp | 0.991855 | 20/20 |
| lj100_local_dz | velocity128 | prefix_prediction | lj_noisy | -14.032257 | 20/20 |
| lj100_local_dz | velocity128 | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_local_dz | velocity128 | teacher_one_step | lj_noisy | 0.520354 | 20/20 |
| lj100_local_dz | velocity128 | prefix_prediction | reid | 0.236453 | 20/20 |
| lj100_local_dz | velocity128 | reconstruction | reid | 0.951530 | 20/20 |
| lj100_local_dz | velocity128 | teacher_one_step | reid | 0.951754 | 20/20 |
| lj100_local_dz | velocity_node64 | prefix_prediction | depablo_low_temp | 0.834438 | 20/20 |
| lj100_local_dz | velocity_node64 | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_local_dz | velocity_node64 | teacher_one_step | depablo_low_temp | 0.991794 | 20/20 |
| lj100_local_dz | velocity_node64 | prefix_prediction | lj_noisy | -10.577249 | 20/20 |
| lj100_local_dz | velocity_node64 | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_local_dz | velocity_node64 | teacher_one_step | lj_noisy | 0.520858 | 20/20 |
| lj100_local_dz | velocity_node64 | prefix_prediction | reid | -0.036492 | 20/20 |
| lj100_local_dz | velocity_node64 | reconstruction | reid | 0.951530 | 20/20 |
| lj100_local_dz | velocity_node64 | teacher_one_step | reid | 0.951770 | 20/20 |
| lj100_observed_acceleration | observed_acceleration | prefix_prediction | depablo_low_temp | 0.976239 | 20/20 |
| lj100_observed_acceleration | observed_acceleration | reconstruction | depablo_low_temp | 0.991859 | 20/20 |
| lj100_observed_acceleration | observed_acceleration | prefix_prediction | lj_noisy | -0.048044 | 20/20 |
| lj100_observed_acceleration | observed_acceleration | reconstruction | lj_noisy | 0.520198 | 20/20 |
| lj100_observed_acceleration | observed_acceleration | prefix_prediction | reid | 0.926420 | 20/20 |
| lj100_observed_acceleration | observed_acceleration | reconstruction | reid | 0.951530 | 20/20 |

Teacher increment R2 across98frames/20validation networks, by [Reid,lowT,LJ]: state32 [.8984,.8678,.8794],state64 [.9191,.9430,.8482],state128 [.9244,.9528,.8346],velocity64 [.9916,.9895,.9856],velocity128 [.9931,.9914,.9848],velocity_node64 [.9919,.9919,.9902]. These aggregated increment scores accompany sourcewise/perframe metrics; smooth-motion baselines remain critical. Step100 teacher p-ratio almost equals persistence and AE. High teacher increment accuracy together with poor free rollouts supports exposure/drift concerns, not proof of causal sufficiency or solved dynamics. Larger models and velocity improved some local metrics without improving LJ free rollout.

Local dz selected epochs state32/64/128120/120/120;velocity6449(of69),velocity12829(of49),velocity_node64113(of120). Acceleration selected237/240; LJ free step100 −.0480 versus AE .5202 (single seed); Reid .9264,lowT .9762. It improves on several local-loss free rollouts but still fails LJ0.4 target; training objectives/budgets differ, so not architecture-only attribution. No new training launched by status request.

### data_mini stiffness short-prefix study — 2026-09-12

User replaces notebook05 input with new data_mini.tar.gz and requests separate node-position/stiffness optimization datasets. Registry audit:669 entries (381node_optimized,288stiff_optimized); sampled stiffness chunk has1500frames,240nodes,689edges, x[N,2], edge_index[2,E], edge_attr[E,4], legacy network.Box. Preserve stored edge channels exactly; no invented angles/rest lengths. Converter retains first50 frames without resampling, original archive and original frame counts retained; fit-only first20 prefix prevents temporal leakage. Conversion metadata/grouping determines deterministic stiffness120-cohort,20train20val80test, no response-based filtering.

Predeclared one2Dseed123MP2/reference16AE (hidden96,32tokens),60epochs/patience12lr1e-4wd1e-5 on frames0..19 only; frozen-AE delta-MLP64/staticmean16 propagators one_step and multistep8,30epochs/patience6lr1e-4wd1e-5,19transitions pertrajectory. Match existing denseReid recipe except new source/counts/horizon and single2Dseed. All normalizers/statistics and checkpoint selection restricted to prefix20; no p-ratio/strain supervision. Both propagators are fitted and frozen before test evaluation. Post-fit validation20/test80 at0..49: physical p-ratio R² from original directional sides quantile0.10, displacement/increment pooledR², validtotal; AE observed-frame reconstruction vs free rollout from0 distinguished. Test20..49 is temporal extrapolation beyond fit prefix as well as held-out networks. No final-test feedback, no seed SD.

Notebook notebooks/dynamics/05_dense_reid_small_steps.ipynb now a compact cached reader for this study, no outputs or execution; preserves early2/3/5 markers and visibly marks training end19, no clipping negative scores. Historical denseReid result artifacts unchanged. Harness hpc/experiments/mini_stiffness; pipeline gates full training behind conversion and2trajectory smoke; no interactive production wait. Exact converter/code/data hashes,cohort IDs and PBS dependencies follow in manifests/ledger.

Conversion sample validation: legacy Box roundtrip preserves x/edge_index/all edge_attr tensors exactly; stored columns0:2 match periodic source-minus-target vectors maxabs0, column2 norm maxabs1.07e-6, column3 stiffness invariant0→1499. Retained frames0..49 all checked for finite data, fixed topology/stiffness; outcome registry remains manifest-only. Group identity sorts edgepairs; one deterministic representative per topology group ensures split separation, requiring at least120groups. Both381node and288stiffness branches retained first50; full original archive preserved.
Archive SHA256 840c8175c83474b4f7c84236a72ed58d9a5e7bfeaedc2bea788ce5bd4b329474; corrected frozen converter SHA256 1d753c1ac24acf0163df7b3bc022a675e5807935b36570cffde43e87d60bbc43; conversion job 4682272.zeus-master. Initial prefreeze4682269 failed on seek in streamed CSV (no data outputs), preserved; superseded4682270 cancelled/finished without outputs, ledger retained. Corrected CSV materialization sample test passed.
Dependent smoke→full training pipeline 4682273.zeus-master accepted afterok:4682272.zeus-master; 8CPU16GB shared mendels_q for conversion and training, no hostpins. Training code manifest SHA256 45b4d4d395351b6085836ee9e0b51ed084e93d9c06125f7cdc527cdf218db332; protocol SHA256 8726822580cb8ff16117ca6df70d9e6023b52ae8162c2b2a858e2e66f1e64064. Dataset conversion still running at submission; full cohort checks/data hashes/prepared.json precede automatic smoke; any conversion or smoke error stops production. No converted-cohort or model-performance claim yet. Notebook cells parse without execution; leave cluster pipeline running.

### Mini-data conversion completed; pipeline metadata repair — 2026-09-12

Conversion4682272 completed both branch first50 files and120 distinct-topology stiffness cohort first20/first50; prepared.json and conversion manifest present. Training4682273 failed in smoke before anyfit: KeyError label; converter emitted source labels at manifest top level instead of nested source. Root moved four fields dataset_name/source_name/label/coordinate_normalization into source, saved prepared_before_label_fix.json, and corrected live converter (frozen historical snapshot unchanged). Failed smoke retained failed_attempts/4682273/d2_s123_smoke. No tensor data changed or reconversion needed. Revised prepared SHA256 c86e1f554a5cf16f4e1a413a66658c812f1fe28c0f7c36a5b4cda098277dda7c. Retry uses same scientific frozen code and existing converted data; no production was trained by failed job.

User clarified canonical mini datasets must retain full1500frames, then capped each branch at200trajectories. Full200conversion is independent of the existing first20/50 experiment inputs. Stiffness200 selection must include existing120 distinct-topology cohort; remaining80 unused by current20train20val80test study. Both branch fullsets use deterministic response-independent selection and preserve original archive/provenance; no truncation to50in canonical files.

Retry4682274 passed smoke fits but failed post-fit reload because resolve_existing_path unnecessarily searched for deleted pyproject.toml even for existing absolute dataset paths. Narrow canonical fix resolves existing absolute paths directly and raises for missing paths; relative-path behavior unchanged. AST-isolated filesystem checks passed; no pyproject restoration. Prior smoke retained failed_attempts/4682274. New frozen code_v2 manifest SHA25605f96d00f26f1b3dcc4b95179d82893dd7423eebf35b5b6f0b29ea9a20c10b23. Pipeline4682276 smoke completed16.54s with bothpropagators/frozenAEchecks and50frameevaluation; full d2_s123 production has begun. Stop production monitoring.

Full manifest clarification: all288stiffness records have1500rawframes and288distincttopologies; all381node records have500rawframes and231distincttopologies. Earlier1500-per-branch wording extrapolated incorrectly from stiffness samples; corrected. User informed that full canonical200 selection will retain200×1500stiffness and200×500node, no padding/interpolation. Choose200 distinct topologygroups for each, with stiffness retaining prior120cohort. Original first20/50experiment unchanged.

Full canonical200 conversion4682277 submitted shared mendels_q8CPU64GB/placefree:shared,24h; conservative memory request for full graph-object retention and torchserialization (~400000frames). Frozen fullconverter includes its canonical-frame helper; exactsource/PBS/archive/prefixmanifest hashes in fullconversionledger. Tiny deterministic selection check:200node topologygroups of231;200stiffness of288, ordered first120 stiffness paths exactlymatch currentstudycohort. Outputs mini_node_optimized_200_500.pt and mini_stiff_optimized_200_1500.pt; manifest mini_data_full200.manifest.json. All actualframes preserved, nointerp/padding; per-frame schema/finite/topology/stiffness checks runoncluster. Fullconversion pending, no completionclaim; existingprefixtraining4682276 proceeds independently.

### Mini canonical conversion and stiffness study completed — 2026-09-12

Full conversion4682277 completed:200node×500frames (~1.5GB),200stiffness×1500frames (~6.9GB); exactcounts/hashes/selection in data/mini_data_full200.manifest.json. Training4682276 completed1133s,peakRSS2.05GiB. Cached final-test results only, no newinference/notebook execution. Single seed123,80/80valid all reportedentries; all fitting/selection restrictedframes0..19. Atframe19 p-ratioR²/displacementR²: AE .5773/.9338,one-step .5466/.9172,multistep8 .5583/.9320. Atframe49: AE .1987/.6733,one-step .1936/.8655,multistep8 .1050/.8527. Here new AE trainedonly20frames also loses response accuracy outsideprefix; one-step rollout almostmatches AE p-ratio at49, while higherdisplacementR² doesnotimplybetterphysicalresponse. No seedSD or final-test-basedmodelselection. Notebook05cachedtables ready foruserexecution.

### Mini stiffness early-frame diagnosis and notebook view — 2026-09-12

User requests p-ratio plot limits0..1 even though negatives hidden; notebook05 now linear0..1 p-ratio panel, physical position/displacement MSE panel=SSE/(2×totalnodes), compact table retains negative R² and validtotal. No new inference/figure export/notebook execution. Cached80test trajectories,seed123,all80/80 response pairs atframes1..5/19. AE p-ratioR² frames1..5 −2.830,−6.346,−6.274,−4.778,−3.037; one-step−2.992,−6.199,−5.761,−4.301,−2.758; multistep8−2.702,−5.223,−4.732,−3.589,−2.432. Thus failure already occurs in AE. AE frame0 residual RMS9.74e-5 versusGTframe1 displacementRMS5.52e-5. Atframe5 AE responsebias−.09060,predSD.01880 versusGTSD.04852,responseRMSE.09687. AE positionMSE frame5 5.45e-9 versusframe19 1.34e-7 while responseR² improves−3.037→.577: small coordinate error alone doesnotestablish goodresponse. Small early deformation, compressed/bias response predictions and narrower GTspread support sensitivity to reconstruction error; no claim of complete causal decomposition without boundary-strain/predicted-coordinate analysis. No new model training or test-based selection.

### Separate node-position optimization study — 2026-09-12

User requests separate node/stiffness notebooks. Preserve completed stiffness05; add05b_node_optimized_small_steps.ipynb and mini_node study. Exact recipe matched to stiffness (2DMP2AE/reference16hidden96tokens32,60epochs/patience12, frozenAEdeltaMLP64one_step/multistep8 30epochs/patience6,seed123; alllearningrates1e-4wd1e-5). Node cohort is ordered first120 of existingdeterministic200 distinctnode-topology subset; use originalfirst50cache mapping byarchivepath for cheap preparation.20train/20val/80test; frames0..19only forfit/stats/selection,0..49postfit. Fullnode200×500canonicaldatasetunchanged; noresponsefiltering, no mixedT/LJ training. Bothmodelsfrozen beforetest. Cohorts differ between node/stiffness studies, so this is not paired causal comparison.

Notebook05b mirrors p-ratio0..1 view and physicalMSE with earlyframes2/3/5 markers; tablepreservesnegatives; noexecution/figureexport. Existing stiffness-specific earlyfailure explanation notcopied to node notebook. Pipeline4682294 accepted8CPU16GBsharedmendels_q/free:shared; prepare→smoke→full. Frozenmanifest SHA25659dd09d25f2325cfc1a5c7b2781021efb70f7f1e26ffaa6f087dc6001643ee9d; protocol SHA256672f5028e5cf24e0f6bdb45d3dda1640b249e2626415d5a75c367ceef2b0d0c1. Exactpreparationpaths/hashes/selectedrecords emittedprepared.json beforefit. Productionawaitssmoke gate; nooutcomeclaim.

Node pipeline4682294 preparation verified120distinctnetworkcohort and20/20/80first20/50boundaries; prepared SHA25624ab01f1f9569f649abd0733fa8a24b0f97eaa37d35eca46b32d52ca88c501b4. Smoke completed16.047s,peakRSS1345860KB,bothfrozenAEpropagators and50framevalidationpassed. Full d2_s123 AEtrainingbegun; stopproductionmonitoring, results pending.

User clarified one notebook for both datasets:05 now has separate stiffness and node sections using shared plotting/loading functions; removed newly created05b. Training/data remain separate and unchanged; p-ratio0..1 and physicalMSE preserved. Cells syntax-checked only, no notebook execution.

### Frozen AE reference-offset diagnostic — 2026-09-12

User requests deeper early-reconstruction investigation; keep2Dlatent/frozenAE. Validation-only20networks each mini_stiffness/mini_node,seed123,frames0/1/2/3/4/5/10/19. Test original decodedpositions, reference-subtraction pred_t−pred_0+true_ref, and translation-only removal pred_t−mean(pred_0−ref) as a response-invariant control. Physical-inverse/originalsidequantile.10 unchanged. Report p-ratioR² validtotal, physicalcomponent MSE, boundary-node MSE, width/height changes/errors, alignment oforiginalframeerrorwithreferenceoffset, and dominant-axis branchswitchcounts. Established estimator uses raw width/height changes and min-magnitude reciprocal ratio, not independently normalized engineering strains; do not silently change it for this diagnostic. Exactreferencezero andtranslation-invariance checks included. No fitting, checkpointselection, finaltest or notebookexecution; cached AE/recipehashes and snapshot/PBS logged. Objective is to distinguish persistentoffset fromchangingearly errors and relativeboundary response error.

Frozen AE audit4682320 completed13s/824880KB,960rows(2sources×20val×8frames×3variants); code manifest SHA2566b4069bf9e344a8b9ada688995748ff503e4ee96f6f7002424f5969ab070a462. Original-vs-anchored pR² at1/5/19: stiffness−1.852/−4.745,−3.947/−5.370,.548/.597; node−29.359/−46.595,−22.422/−24.542,.327/.345; every20/20valid. Translationcontrolpreservedresponse, no originalbranchswitches at1/5/19. Exact parity ofneworiginalp withcachedvalidationresults (maxdifference0); physical-inversemaxabs 3.183085937052965e-07. Reference subtraction helps somepositionMSE butworsens earlyp, so notsimpleoffsetfix. Frame5meanboundaryratio(pred/true):stiffnessy.628,x1.154;nodey.487,x1.415. Underpredictedtransverseexpansion andexcesscompressionexplain lowpredictedp directly; underlyingtraining/representationcause notuniquelyestablished. Allmodelsfrozen/no newtraining/notebookoutputs;fulltables/README in mini_ae_audit.

### Frame-balanced coordinate AE fine-tuning pilot — 2026-09-12

User authorizes bestnextstep under strictnoexpertobservablesupervision/selection rule. Fourshortfits: twoexisting2D AEs(mini_stiffness,mini_node)×uniformcontinuation/balancedframes,seed123. Frozenstartingcheckpoint+normalizers; AEweights fine-tuned, architecture2DMP2/reference16 unchanged. No posthocreference subtraction (previousaudit worsenedearlyp). Training20networks×frames0..19,validation20only,no finaltest. Frameenergy is training-only per-network mean squared normalized displacement divided byfrozen targetstd; flooredat.04medianpositive-frameenergy, inverseweights normalizedmean1. Frame0 includedwithfloor. Objectiveandcheckpointselection both weighted per-graph reconstructioncoordinateMSE; uniformcontrolweight1. Adam1e-4wd1e-5clip1,30epochs/patience8,batch32,minimprove1e-8,epoch0eligible. Normalizers fixed; p-ratio/strain absentfromweights/features/loss/selection. PostfitAEpR²,physicalMSE/R²,validtotalat0..19,sourcewise single-seed. No propagatorchange/notebookexecution. ProtocolSHA256 9b10077d49906a42aadd9a5bc1e33e749222198744c714ffe27d8a65aa165078. Smokesgateproduction; scientificrunner hpc/experiments/mini_ae_frame_balance/run.py.

Frame-balance immutablecode_v1 manifest SHA256f572e56d43cf7cf313cc36838cdbff335d7e7e8c81f05ef3fe58e7bbbb82787f. Smokes4682323(stiffnessuniform),4682324(stiffnessbalanced),4682325(nodeuniform),4682326(nodebalanced) submitted shared4CPU4GB; fullfitsgated oncompletevalidationevaluation andnormalizer/originalcheckpoint invariants.

Allfour frame-balance smokes passed exit0:finitepositive20weights, energyfloorsstiffness.0351915/node.0342468, unchangednormalizers/originalcheckpoint,40pernetwork/20perstepvalidationrows. Runner1.785–2.059s,PBS10–11s,RSS1035280–1165084KB. Production4682327stiffnessuniform,4682328stiffnessbalanced,4682329nodeuniform,4682330nodebalanced acceptedRunning4CPU4GB/shared. Exactcommands/IDs/dependencyhashes in mini_ae_frame_balance/job_ledger.json. No productionwait or resultclaim.

### Frame-balance pilot completed — 2026-09-12

Allfour production4682327–4682330 completed, nofailedartifacts; everyarmselectedepoch30of30 (budgetcap,notconvergenceproof). Originalcheckpoint/normalizers unchanged. Cachedvalidation20/20validperreportedframe,seed123,no finaltest ornewinference. p-ratioR² atframes1/3/5/10/19: stiffnessoriginal[−1.852,−8.078,−3.947,.562,.548],uniform[−2.993,−1.612,.267,.945,.903],balanced[−1.974,−.254,.633,.947,.850]; nodeoriginal[−29.359,−41.803,−22.422,−1.244,.327],uniform[−23.131,−2.836,−5.305,.617,.842],balanced[−39.395,−1.139,−1.299,.652,.599]. Stiffnessframe5/19physicalMSE original4.075e-9/1.245e-7,uniform1.439e-9/3.083e-8,balanced7.768e-10/4.409e-8; nodeoriginal1.572e-8/7.418e-8,uniform2.382e-9/4.353e-8,balanced1.236e-9/6.753e-8. Extra training alone improves substantially; framebalancing improvesframe5againstuniform buttradesoffframe19anddoesnotresolveframe1/nodeearlyresponse. No blanketbestweighting/convergenceclaim. No propagatorfits/notebookchanges/newjobs fromstatusrequest.

### Expand mini investigation to100frames — 2026-09-12

User requests moreframes andcontinuedinvestigation. Extend both fitting andevaluation toexactframes0..99, maintaining original20/20/80identities and2Dlatent. Dataprep4682337 submitted8CPU64GBshared tosliceexistingfullcanonicaldatasets sequentially; no resampling, exactfirst120cohortchecks. Source100preparedmanifests retain full/prefix/codehashes.

FourAEarms2sources×uniform/balanced allstartfromsameoriginal20frameAE per source, not mode-specific30epochfine-tunes. Keeporiginaltraining-onlynormalizersfixedforcomparison;60epochs/patience12Adam1e-4wd1e-5clip1batch32. Balanceweights use100training-frameenergies withfloor .04medianenergyframes1..19, retainingprior earlyfloor; normalizeinverseweights mean1 over100frames. AEcheckpointselection matchingweightedcoordinate validationloss only. Noresponse/strainfeatures,weightingorselection. EachAE thenfrozenfor two latentdeltaMLP64/staticmean16propagators(one_step,multistep8),99transitions/source,60epochs/patience12lr1e-4wd1e-5,selectvalidationlatentloss. All8propagatorspredeclared. PerAE bothpropfitsfreeze before80testevaluation; allAEandfree-rolloutphysicalp-ratioR²/MSE/R² at0..99withvalidtotal,one seed noSD. Previouslyinspectedtestcohort is heldoutdevelopment, notfreshblindtest; no feedbackduringfitting. Budgetandlengthincrease together; historicalcomparisonnotisolatedcausalframeeffect.

ProtocolSHA256 19fc12a7a2d8a6d63d46db7bbb720e1b28db0c0a23997a8272d947f9d20b2646. Notebook05now100-framecachedreader,sameone-notebooktwo-datasetstructure; separate ordinary/balancedcolumns, pR²0..1/MSE, early2/3/5markers. Noexecution orfiguresaved; oldresultsretained. TrainingpipelinesgateAE+bothpropagatorssmoke beforefull; results pending.

100frame productionpipelines4682342(stiffnessuniform),4682343(stiffnessbalanced),4682344(nodeuniform),4682345(nodebalanced) accepted8CPU16GBsharedafterokprep4682337. Frozenmanifest SHA256433fa444d79375991148e729d12ccb487c23c024f8ff242333e96f202025a050; protocol SHA25619fc12a7a2d8a6d63d46db7bbb720e1b28db0c0a23997a8272d947f9d20b2646; job_ledger.json exactcommands/deps/hashes. Allfour automaticAE+bothpropagator smoke gates passed through99 (200perstep rowsperstrategy, AEunchangedassertion); fullproductionbegun. No failures, no productionwaiting; newmetrics pending.


### Mini100 one-step-only scope and shared plotting theme — 2026-09-12

User retired the 8-step propagators from the current study. At the operational check, jobs4682342–4682345 were already terminal F/Exit_status0 with all four one_step completion and test CSV artifacts present; no cancellation or retraining was needed. Live mini_100 propagation and smoke checks now require only one_step. Frozen code_v1, original protocol, and historical multistep8 results are preserved; exact operational changes are recorded in `notebooks/results/mini_100/operational_change_20260912_stop_multistep8.json`. Notebook05 now compares AE reconstruction with one-step autonomous rollout only, retaining both datasets and both AE weighting arms. No new metric analysis or model selection in this change. Shared `lss.plotting.apply_editorial_style(mode)` supports light/dark; active dynamics and engineering notebooks expose PLOT_THEME at the beginning. Lightweight theme rendering and code-cell syntax checks passed; no notebook execution or figure export.


### Mini shared originals and mixed-temperature transfer — 2026-09-12

User requests fresh2D MP AE+one-step propagator fits and a clean notebook with single-network demonstrations, shared source-wise responseR²/MSE, latent z0/z1 associations to p-ratio and physical x/y boundary deformation, and matched unseen-network comparison of stiffness-only versus joint-original fits. Explicit confirmation: train only originals; mixed-T excluded from normalization, fitting and selection. No expert observables in inputs/loss/selection. All200frames0..199 used without resampling. Four predeclared arms single_stiffness/single_node (train identity0 only, validation20..39), stiffness_only (20stiffness originals), joint_original (20stiffness+20node originals); validation20/source, test80/source identities40..119, unused120..199. One seed123; prior-inspected held-out development, not freshblindtest. Equal epochs/countper-source, joint has moregraphs, not a controlled total-data causal comparison.

Architecture MP2/reference16,2Dlatent,hidden96,tokens32. Fresh initialization then1ordinaryAdamW warmup epoch and60maximum frame-balancedAdam epochs/patience12,lr1e-4wd1e-5batch32clip1; source-wise train-only inverse normalized displacement energy withfloor .04medianframes1..19 and source-mean-one weights. Select mean weighted validation source coordinate MSE,min1e-8. Frozen training normalizers afterwarmup. FrozenAE propagator deltaMLP64/reference-mean16, no source/temperature input, one-step latentdelta MSE,60max/patience12lr1e-4wd1e-5,199transitions; val latentlossselection. No8-step variants. Allfits frozen before productioncollectors. Exactrecipe `hpc/experiments/mini_shared/protocol.json`.

Dataaudit4682393 succeeded: four200×200files, allfinite coordinates, paired topology equal atall200frames for200pairs/source, first120node/stiffness topologies distinctwithzerooverlap. Unmodified trainval40/test80caches, hashes and temperatureassignment/MDprotocol in `notebooks/results/mini_shared/data_manifest.json`. Mixed T={.1,1,5,10,20,30}K,counts34/34/33/33/33/33,assignmentseed904297,200MDsteps/frame,dt.01,compressionincrement2e-5,noLJ. Originalsourcechunk/ID metadata retained; mixed identities established via source_index plus topology. PeakRSS3070212KiB. Audit4682391 failed on missingmixedsource_chunk assumption beforecaches;4682392 completedaudit;4682393 finalizedcollectorcontract andcrossbranchoverlap. Failures/logs retained.

User additionally requests quick single-network transfer: eachfrozen singlemodel evaluated on matchedmixedID0 (response trace, no across-networkR²), and20unseenoriginal/mixed identities40..59 (pR²/MSEwithvalidtotal). This separates same-network temperature transfer from unseen-network generalization; no extra fitting. Postfit physical endpoint p-ratio estimator fixedframe0sidesquantile.10 preserved; physical inversion checked. Latent coordinate associations descriptive, not intervention claims; raw axes across AEs arbitrary rotation/sign/scale. New notebook `notebooks/dynamics/06_mini_shared_latent_dynamics.ipynb` reads cachedresults only, sharedlight/darktheme, pR²ylim0..1. Notebooknotexecuted; training/evalresults pending.

Mini_shared submission confirmed: fitting pipelines4682399(single_stiffness),4682400(single_node),4682401(stiffness_only),4682402(joint_original); collectors4682403–4682406 respectively, eachafterok ALLfourfitjobs. All8CPU16GBsharedmendels_q/free:shared. Frozen code_v1 manifestSHA256459cf061ce9f6ce0e3149d7c6c2a1229ddbbda91ae209bf11edbcef7e60d59bc; protocolSHA2563c2259382e905c968cd9513b60db2ecc9bf850754d44e6507cb84fed68b5e617; data_manifestSHA25622c1885f87a79260b7c7cf2a2d9040f04165dc0f08ee9d0125e6f3029dd78cae. Exactcommands/dependencies/status in job_ledger.json. Firstboundedcheck: all4fitPBSrunning, first3smokefitscomplete, jointsmokeAEcomplete/propagatorprogressing; noobservederrors. Productiongatedonsmokeevaluationcompletion. No fulltrainingwait or outcomeclaim.


### Additional all-four mini training arm — 2026-09-12

User explicitly authorizes joint_all as an additional comparison if original-only transfer fails. Original-only arms and frozen code_v1 remain unchanged. New arm includes stiffness_original,node_original,stiffness_mixed,node_mixed with20train/20validation/80test trajectories per source, identical paired-network split boundaries and seed123, frames0..199. Same2D MP AE/frame-balanced coordinate loss and frozen one-step latentdelta propagator recipe; no temperature/sourceID input or expertobservable supervision. Training-only normalizers and validation selection now include mixed sources for this arm only. Therefore joint_all mixed results are held-out network performance with mixed-T training exposure, not zero-shot temperature transfer. More source trajectories imply greater training exposure at the same epoch cap; no exposure-matched causal claim. Newcode_v2 and separatejobledger preserve originalrecipe/provenance. Notebook06 MODEL switch supports joint_original or joint_all; noexecution or resultsclaim.

Joint_all fitting4682407 accepted/running; collector4682408 heldafterok4682407.8CPU16GBsharedmendels_q/free:shared. Immutablecode_v2 manifestSHA25626a82b1596f6a6052107cb6b9d077f873e3a0eb21ad7f27f143c7ba997b01acc; protocolSHA25611d3b9c73a80166384a1a8dcb6873db074cf677432becfa21d2963795d8a2fbd; dataSHAunchanged22c1885f87a79260b7c7cf2a2d9040f04165dc0f08ee9d0125e6f3029dd78cae. Separate joint_all_job_ledger.json contains exactcommands/dependencies. Initialjoblogempty, smokeoutcome notyetknown. Existing4682399–4682402 stillrunning anduntouched. No productionwait or metricclaim.


### Mini shared completed results — 2026-09-13

Allfivearms fit_completed and completed artifacts present. ReadcachedCSVsonly; no newinference/notebookexecution. Source-wise horizons99/199, AE/rollout, validtotal andsecondarymetrics collectedin notebooks/results/mini_shared/brief_20260913.csv. Seed123single-run, noSD. Heldout80/80valid each reported main source/horizon. Joint_original at99 AE/rolloutpR² stiffnessoriginal .987896/.832477,nodeoriginal .981239/.950737,stiffnessmixed −.010705/.088865,nodemixed .085782/.108923. At199 stiffnessoriginal .971782/.833843,nodeoriginal .963663/.922629,stiffnessmixed −.086968/.057466,nodemixed .094998/.332096. Thus strong original-source AE and rollout, weak mixed transfer under frozen endpoint estimator. Stiffness-only at99 AE/rollout stiffnessoriginal .984690/.832050,nodeoriginal .731336/−4.417131; adding node training substantiallyhelpsnode rollout. Joint_all AE androlloutpR² nearzero/negative onallfour (range−.149to.006at99), so addingmixed didnotfixthisrecipe anddegradedoriginals; causeunresolved, notproofarchitectureincapacity. Single-network fits donotgeneralize to20unseen originals/mixed: all99AEandrolloutpR²negative,20/20valid. Same-network pairedtransfer isdistinct andnotassessed by thisbrief. Mixed scores here are savedphysicalendpoint fixedframe0sidequantile.10 p-ratio, not across-frame thermalregression; do not conflateestimatornoise withirreducibledynamicsfailure. No newmodelselection or traininginitiated.


### Mixed-T response corrected to trajectory linear fits — 2026-09-13

User explicitlyrequests mixed-T p-ratio from fulltrajectory linearfit toreduce thermalnoise. Postfit-only reevaluation ofexistingfivefrozen mini_shared arms; no retraining/normalizerchange/checkpointselection. New thermal_response.py matches existing thermalengineering frame_time_ols_p_ratio: fixedframe0q.10boundarygroups, physicalcoordinateinversion, width/height normalizedbyfirstfinitevalue, twoOLSstrain-vs-frame fitswithintercepts,p=−slope_y/slope_x,>=8finiteframes andabsolute fittedaxialtotalstrain>=1e-3. Undefinedprefixes retainedasNaN withvalidtotal. R²curveat t usesonlyframes0..t;199usesall200. Mixedlatent p_ratio usesfulltrajectoryfit (retainsendpointcolumn); x/ydeformation staysframewise. Original endpoint metrics reusedunchanged. These estimators have explicitlydifferent normalization/branchdefinitions; cannotattributeallscorechange solelytosmoothing.

Oldendpoint artifacts andbrief conclusions preservedashistorical. Refreshedtree notebooks/results/mini_shared/thermal_fit; recipe.json describesestimator/tests. Notebook06nowreadsthisnewtree; noexecution. Lightweighttests: exactagreementexistingthermalmetric at1/8/30/99/200frames, analyticpositive/negative response, flatcompressionundefined, changedfutureframesdonotalterearlierprefixfit. Need frozencheckpointreevaluation forpredictedwidths/heights; existing CSVshadonlyendpointp,insufficienttorecovertrajectoryfits. BulkreevaluationonsharedHPC, outputs/metrics pending.

User requested existinggraph_utils implementation. Checked installedpackage andfreshGitHubmain+remoteheads: both734c6b81b5cddc539d1fad2e465e333f45ee5a90, onlymain, endpointhelpers only. No packageupgrade needed. Removed duplicatefitmath fromnewadapter; delegate directly toexisting hpc/experiments/ml_only_engineering/thermal_metric.py::frame_time_ols_p_ratio, snapshottedverbatim/hashpinned inreevaluationcode. Adapterrepresentscachedwidth/heightasfourboundarymeans, mathematicallysufficient forsamefunction. Directscientific-environmentimport andanalytic/undefinedprefixchecks passed. No changednumericalestimatordefinition; newjobsstillpostfitonly.

Thermalresponse refreshsubmitted:4682602single_stiffness,4682603single_node,4682604stiffness_only,4682605joint_original,4682606joint_all; evaluationonly, no8-steporothertraining. Snapshot thermal_fit/code_thermal_fit_v1 manifestSHA256d6e854751490f2ee3d05bddfbf52e62ab52823cb4bc74e8faede3ebf8d9680df; existinghelperSHA256e4b2a6af2ba9927a5ff8547eee58b34457674941b1efea2dcfa714cefba910bd; exactjobs/hashes inthermal_fit/job_ledger.json. Strictcohort/endpointprediction/displacementidentitychecks appliedincludingpairedmixedID0 fromtrainvalcache. Outputs pending.

Notebook06 adds absolutePearson/Spearman heatmaps bysource atuser-chosen CORRELATION_FRAME(default99),2latentrows×3observables,shared0..1scale,finitepaircounts andundefinedconstantcells. Eachnetworkcontributesonce perframe/source; avoidsconfoundingpooledtimeprogress. Signedcoefficientsretainedinheatmap_correlations; no p-values or modelselection. Removedlargeperframecorrelationtabledisplay, retainedscatters. Syntaxcheckedonly; notebooknotexecuted.


### Thermal refresh launcher recovery and notebook paths — 2026-09-13

User reported missingthermal_fit/single_stiffness/demo/per_network.csv. Logs4682602–4682606 showargparseinvalidchoiceemptyarm: launcher passedLSS_ARM empty, allfailedbeforeinference/output. No oldresults/checkpoints modified. Recovered evaluation-onlyjobs4682612single_stiffness,4682613single_node,4682614stiffness_only,4682615joint_original,4682616joint_all bindarms explicitly. Eachcreatedrefresh_recipe.json confirmingnonemptyarm andcorrectfrozenmodelpath; fullcompletionnotyetclaimed. Failedsnapshot/attempt and recoverykeptinthermal_fit/job_ledger.json. Notebook06originaldemoreads ORIGINAL_RESULTS directly; thermalplotsseparatedintosubsequentcell withexplicitpending-evaluationnote. Existingoriginaldemofiles andallcodecellsyntaxverified; no notebookexecution/fallbacktomislabelledendpointthermalresults.


### Cached latent geometry interpretation — 2026-09-13

User requests concise paper interpretation, nojobs/notebooks. Inspected same80heldout original networks/source acrossstiffness_only andjoint_original, seed123all200frames. PertrajectoryPCA inraw2Dcoordinates: medianPC1variancefraction node .995598→.999440, stiffness .998896→.999570. PrespecifiedexampleID40 node .996096→.999460; stiffness .999742→.999712. Saved exactrows inmini_shared/latent_geometry_comparison_20260913.csv. Bothrepresentations trace predominantly1Dpaths; jointdoesnotshowincreased2Dspreadasnecessarybenefit. CoordinatePCA notinvarianttoarbitraryrescaling andnotintrinsicdimensionproof. Cachedframe99AEpR²node .731336→.981239, rollout −4.417131→.950737; stiffnessAE .984690→.987896,rollout .832050→.832477,80/80valid. Supports broader predictivecoverage inthissingle-seedcomparison; cannotisolate latentgeometryfromdecoder/staticcontext/propagator retraining or increasedtrainingexposure. No claimdisentanglement or mechanisticcausality. PrimaryliteraturecheckedKornblithetal2019 representationcomparison (PMLRv97): representationcoordinateshaveinvariances,rawaxisdifferencesnotaloneinformationgain.


### Mixed linear-fit failure diagnosis at frame100 — 2026-09-13

Verified refreshed joint_original/joint_all completed artifacts andcachedprefixfitdiagnostics; no inference. Atframe100 all80mixed trajectories eachhave101finiteframes inGTandpredfit. Thusfituses0..100, notendpoint. joint_original node_mixed AEpR².834272 butonly10/80validpairs; stiffnessmixed AE−.088317with19/80. GTvalid76/80node,71/80stiffness. MedianfittedaxialstrainGT node−.00205631/stiff−.00201891, AEpred−.000254831/−.000053806; muchweakerdecodedcompression causespredminimum1e-3axialstrain gate torejectmost. Do notinterpret highnodeAEconditionalR²asgoodfullcohortreconstruction; changingvalidsubsetcanmakecurvesirregular. Rolloutvalid76/80node,71/80stiffness,pR² .049244/−.013706. Rolloutmedianaxialslopes−2.00977e-5/−2.28129e-5 nearGT−2.05631e-5/−2.01891e-5 buttransverseslopedistributioniscompressedrelativeGT. Linearaveragingdoesnotcorrectbiasedpredicteddeformation anddoesnotguarantee eliminationofslowthermalvariation. Originalendpointchecks/refreshreceipts preserved; noadditionaltrainingfromthisquestion.


### Ground-truth mini original/mixed data notebook — 2026-09-13

User requests paired200frame ~3sGIFs andp-ratiohistograms overallnetworks. Data-onlypreview4682624 completedpilot+full~116s,shared8CPU16GBmendels_q/free:shared; prior4682623failedonGIF filename-extensionencodingbeforecompletion,preserved. No models/fitting/selection. Newnotebook07_mini_data_original_and_mixed_temperature.ipynb reads notebooks/results/mini_data_preview/{networks.csv,frame_series.csv,metadata.json}.800networkrows,200persource(node_original,node_mixed,stiffness_original,stiffness_mixed);160000framerows. Samefixedframe0q.1sides endpointandexistingthermalOLSfull/prefixestimators appliedtoALLfourphysicaldatasets, includingvalidcounts; comparabilitynotes explainrawendpointvsnormalizedstrainfitdefinitions. Metadatarecordsfullinput/helper/codehashes,widthvsboxdiagnostics,resourceusage andmixedrestlength-rebuild/zeroangleprovenancecaveat.

FourpairedGIFs preselectedfirstsourceindexatT.1and30K(categorynode/stiffness,IDs0and24each), withoutresponsefilter; savedalongside notebook in titlefolder Mini data — original and mixed temperature. Sharedphysicalscales, periodicminimum-imagesegments, noexaggeration. Rootindependentlyverifiedall4files200framesand3000ms andeachselectedsourcecurve200rows; CSVsourcecounts200each,originaltemperatureblank(actualTnotrecorded). Agentpostprocessedlabels mixed_temp→mixed andclearedmistakenpaired-temperaturemetadataonoriginalrows withoutrecomputation; preservedcorrectionrecord. Notebooksyntaxchecked, notexecuted; nofiguresexportedapartfromrequestedGIFs. Outputsreadyforuser.


### Mixed-T historical p-ratio estimator parity — 2026-09-13

User requested the same mixed-T response estimator as the earlier successful
06b transfer study. The historical0.901 original checkpoint, exact code and
30-trajectory split remain unavailable, so it cannot be exactly replayed.
The preserved five-seed2D reconstruction was independently reaggregated from
raw per-network frame100 rows: mixed-T autonomous endpointpR²
0.9041634848±0.0115603800,20/20valid per seed3456456/123/456/786/2026.
All stored primary and explicitly labelled endpoint columns agree.

Notebook06 endpoint values were independently recomputed from cached physical
fixed-frame0 q.10 side widths/heights using the historical symmetric minimum-
reciprocal displacement formula. NaN patterns agree across64,000 mixed rows;
maximum absolute disagreement at reported frames99/199 is2.074e-13 (the
all-frame maximum8.637e-7 occurs near zero deformation at very early frames
from CSV roundoff). Thus the preserved endpoint metric is numerically
equivalent for the cached positions. The full-trajectory normalized-strain OLS
metric is not equivalent and remains a separately labelled auxiliary
diagnostic.

Under the like-for-like endpoint estimator, joint_original mixed-T frame99
AE/rolloutpR² is0.085782/0.108923 for node and-0.010705/0.088865 for stiffness;
frame199 is0.094998/0.332096 and-0.086968/0.057466,respectively,all80/80valid.
Therefore estimator mismatch does not explain the failed recovery on the new
small-step data. Notebook06 now reads original endpoint caches for its primary
mixed-T plots; stale OLS outputs were cleared and code syntax checked, but the
notebook was not reexecuted. Audit script/result:
`tools/analysis/audit_mixed_pratio_estimator_parity.py` SHA256
593f5ad6780054a7130d2b5842f0b3325fd759862bb012c2c920a8e79b4391ca and
`mini_shared/mixed_pratio_estimator_parity_20260913.json` SHA256
acf575d055a0858b878fa1b5605780c126e5218bd8e14ff1ec0b729f3fef4dee.
No inference, training, checkpoint selection or final-test access.


### Uniform-frame mini shared and low-T-only extrapolation pilots — 2026-09-13

User requests evaluation at the 200th stored frame (frame199) with the
historical physical endpoint estimator, while removing the large inverse-
deformation early-frame emphasis. Two single-seed123 pilots use all frames
0..199 and exactly unit post-warmup weight for every training frame/source.
Architecture, 2D latent, normalizers, optimizer, 60epoch/patience12 budgets,
and frozen one-step delta propagator match mini_shared/joint_original. Model
selection remains uniform validation coordinate MSE and validation latent-
delta loss only; p-ratio/strain never enter training or selection. Mixed-T is
excluded from fitting, normalization and selection and is evaluated only after
the checkpoints are frozen. Primary post-fit reporting is source-wise AE and
autonomous-rollout p-ratio R2 at frame199 using physical coordinates and fixed
frame0 q.10 sides, with valid/total counts. Test80 is previously inspected
held-out development data; each one-seed result is preliminary with no SD.

Joint-original uniform retry2 trains20/validates20 for both
stiffness_original and node_original; fit/smoke job4682637 and dependent
collector4682638 are accepted (mendels_q,8CPU,16GB,placefree:shared). Frozen
code_uniform_v3 manifest SHA256
3ffdedb155ece2c339bdb7d59d0513b2693bb066ac4067c7e6c7b6102ce615ae;
protocol SHA256
42ad7396e411b73ae8e3e8f903a005bf710e342d462043da3e3ed2902a138116;
exact provenance/status in mini_shared/uniform_retry2_job_ledger.json.

Low-temperature node-only retry1 trains20/validates20 node_original networks
and evaluates exactly node_original and node_mixed test80, directly testing
dePablo/node-optimized low-T to mixed-T extrapolation without stiffness-source
training. Fit/smoke job4682639 and dependent collector4682640 are accepted with
the same resources. Frozen code_node_uniform_v2 manifest SHA256
558ab83e6ebf0e9bb169e7d939458f22200866a2b109ff93b14e45a0e2303356;
protocol SHA256
587c1443f3f3b4329b1237b6d5c16d6f5c8edbb94c607dd711f0383f7eb90e63;
exact provenance/status in mini_shared/node_uniform_retry1_job_ledger.json.

Failures are retained. Initial joint job4682630 failed in smoke because the
parity assertion compared intentional smoke propagator limits1/1 against
production60/12; collector4682631 dependency-failed. Joint retry1 job4682633
passed smoke fit/evaluation but failed before production because PBS logging
pre-created the protected result directory; collector4682634 dependency-
failed. Initial node job4682635 was terminated before completing smoke when
the same logging collision was identified; collector4682636 dependency-
failed. Corrected launchers keep logs outside result directories and preserve
strict no-overwrite creation. Both final pilots were running and collectors
held afterok at the independent status check; no production metric claim yet.

Both corrected pilots subsequently completed with all four jobs exit0.
At frame199, joint-original uniform AE/rollout endpoint p-ratio R2 was
node_original .964865/.950210, node_mixed -.026687/.387178,
stiffness_original .978978/.836946 and stiffness_mixed -.302444/.035497;
all entries80/80valid. The AE selected post-warmup epoch9 (stopped after21),
and the propagator selected epoch58 of60. Fit4682637 used3:25:40 and
2589592KB; collector4682638 used00:17:09 and2231656KB. Checkpoint SHA256:
AE54356df4072e7d8a5fa9457ef27a762f3a1e0a61afa2b934d44fc8587e1b5a3b,
propagator0880795f4157ff1efc8f70a16025afc949eb4171b125f48aca1519ff899e5f6f.

Low-T node-only frame199 AE/rollout endpoint p-ratio R2 was
node_original .970263/.961533 and node_mixed .082617/.362967, all80/80valid.
The AE selected post-warmup epoch13 (stopped after25), and the propagator
selected epoch55 of60. Fit4682639 used1:29:01 and1651224KB;
collector4682640 used00:08:02 and1522348KB. Checkpoint SHA256:
AEf6bddc23488c3518356419e4fefdb3ce8df5b2c99bf43ec29e27a0168fd47275,
propagator40541ce146e9d58ad6aa5a177589cee5fbc25a8c8ece446d74adb76a0a2647e3.
Uniform weighting therefore did not recover the historical mixed-T transfer.
Against the prior joint_original frame199 endpoint rollout, it modestly raised
node_mixed .332096→.387178 but lowered stiffness_mixed .057466→.035497;
AE response reconstruction worsened for both mixed sources. Node-only training
gave node_mixed rollout .362967, also far below the reconstructed historical
.904163 five-seed validation result. These are single-seed, previously
inspected held-out-development results, not untouched-final-test evidence.


### Matched 1,000-MD-step mini-temperature reruns — 2026-09-13

User identified four newer MetaForge datasets intended to test larger stored
deformation steps. Full audit of all160,000frames passed: four datasets each
contain200trajectoriesx200frames, times0..199000 exactly every1000MDsteps,
finite coordinates, monotone exact1.99% axial compression, and fixed topology
and edge stiffness within every trajectory. Low/mixed pairs preserve all200
source_index/source_chunk identities and use the same per-index thermal seed.
Low uses1e-7K; mixed uses0.1/1/5/10/20/30K with counts34/34/33/33/33/33.
Dataset SHA256: node_low fa9413516e252825a9e93f73f87edd1e361004807c6fe158e6b54ace1678784b,
node_mixed d4f85c76434470293c75968f7c9bd29b9ff55f76012c9fffbb73f9f9b3ee49f9,
stiffness_low 5cdb873ac50118a65294f6ca0db9e3f25a22f6f0ed30fa089951b0e521e1c1f8,
stiffness_mixed 211e9f5738b1abcaacfff4295400c5b2812983c9f299c020aca2fab0ec2f790c.
Matched mixed-vs-low endpoint p-ratio R2 across all200 is.887131node and
.712079stiffness,200/200valid each; high-temperature stochastic response is
larger, especially stiffness20/30K. Historical-source comparisons have
different horizons (nodeabout1%, stiffnessabout3%) and do not invalidate the
internally matched1.99% experiment. These simulations rebuild rest lengths
from each source first graph because the original archive lacks them; this is
a new matched physical condition, not an exact historical-trajectory replay.
The prior uniform results used old times0..39800 stride200 and therefore did
not test this hypothesis.

Two seed123 reruns use direct new files with trainIDs0..19,
validation20..39 and previously inspected heldout-development40..119. Both
use all200frames, exact unit post-warmup AE frame weights, the matched fresh2D
MP AE and frozen one-step latent-delta propagator,60epochs/patience12, and
coordinate/latent validation selection only. Mixed data is excluded from all
fitting, normalization and selection. Primary post-fit metric remains physical
fixed-frame0 q.10 endpoint p-ratio R2 at frame199, separately for AE and
autonomous rollout with valid/total. Joint low-T fit4683622 and collector
4683624(afterok) cover node+stiffness low training and allfour evaluations.
Node-only low-T fit4683623 and collector4683625(afterok) evaluate node_low and
node_mixed only. All jobs accepted on mendels_q,8CPU,16GB,placefree:shared;
fits running and collectors held at independent check. Frozen code_v1 manifest
SHA25605b6ca2efa2e12d7cd608569c4e3645b74b6cc345f35b9ae20cb1dc0bddc0bb8;
data manifest SHA256fcf9ca1a95e81aaa8a13ab098d91d611163cbdf10df74b2ac0d8ea06de4bebb2.
Frozen static AE/propagator parity assertions passed for both smoke and
production budgets before submission and were independently repeated after
submission; runtime also asserts frame time=1000xframe. Exact paths, hashes,
jobs and status are in mini_temperature1000_uniform/job_ledger.json. Results
pending; single seed means noSD and no final-test claim.

Initial fit jobs4683622/4683623 subsequently failed after successful AE and
propagator smoke fitting, during smoke endpoint collection only: direct
MetaForge graphs expose source_index but not the old-cache source_sim_id field.
Collectors4683624/4683625 terminated by afterok dependency. No production
training or test evaluation occurred. Failure logs and smoke artifacts remain;
the original ledger is marked failed. This is an evaluation identity-schema
failure, not a model or dataset result.

Schema-only retry assigns the declared smoke validation identity20 explicitly
before evaluation and requires returned trajectory_id={20}; production still
assigns and asserts exact IDs40..119. No data, model, objective, split, seed,
weighting or estimator changed. Joint retry fit4683633 and collector4683635
(afterok); node-only retry fit4683634 and collector4683636(afterok). At an
independent live PBS check both fits were stateR and both collectors stateH,
with mendels_q8CPU16GBplacefree:shared. Frozen code_v2 manifest SHA256
94cb298e078e24e3e75fbaccf848e0f39f7af10bca079f53f59a07d40e854766;
unchanged data-manifest SHA256
fcf9ca1a95e81aaa8a13ab098d91d611163cbdf10df74b2ac0d8ea06de4bebb2.
The retry submitter itself runs smoke/production configuration parity before
freezing/submission. Exact status is in
mini_temperature1000_uniform_retry1/job_ledger.json; results pending.

Corrected1,000-step reruns completed: fits4683633/4683634 and collectors
4683635/4683636 all exit0. Cached frame199 historical-endpoint p-ratio R2,
AE/rollout respectively, all80/80valid: joint node_low .952517/.958657,
node_mixed .745768/.883940, stiffness_low .927525/.835224, stiffness_mixed
.482135/.566453; node-only node_low .959749/.960070 and node_mixed
.529379/.873666. At frame99 joint node_mixed .561122/.645147 and
stiffness_mixed .254239/.421346; node-only node_mixed .403058/.612940,
all80/80valid. Thus the five-times-larger stored step/1.99% horizon materially
improves mixed-T response recovery. Joint node_mixed rollout rose from the
old-stride uniform .387178 to .883940; stiffness_mixed .035497 to .566453.
Node-only mixed rollout rose .362967 to .873666. This strongly supports the
larger-signal hypothesis for node transfer and partially for stiffness; it
does not isolate stored cadence from the simultaneously longer total strain.
The node result is close to, but not an exact comparison with, historical06b
.904163 because data generation, horizon, cohort and seed aggregation differ.

Joint AE selected post-warmup epoch6/stopped18; propagator selected20/stopped32.
Node AE selected13/stopped25; node propagator selected60/stopped60, so the last
is budget-capped rather than convergence-proven. Checkpoint SHA256 joint
AE6724f375c0f23781647631e5e9d07c3e92807e772f8a3eb89d2b626b24583040,
joint propagatorfe3785ca90b192da783eda59e94c2f8ee31f89f74785e1125c11a0e13179ea6f,
node AEf22ebf56c232e269c3466f38fa40706a045499b0601baff69476ef99413d92bd,
node propagator0547d98ceb07aec5235ea671c81358a6f9f8e4d172095a0c41062ad159b30e44.
Walltime/maxmemoryKB: joint fit02:29:32/3491624, collector00:18:54/2813884;
node fit01:38:23/2060780, collector00:09:43/1849116. All are single-seed123 on
previously inspected heldout-development IDs40..119; noSD/final-test claim.

### Mixed-T 1,000-step linear-response evaluation — 2026-09-14

User requested suppressing mixed-T thermal fluctuations with a trajectory fit.
This is post-fit evaluation only of the frozen corrected 1,000-step checkpoints;
there is no retraining, normalization change, response-based selection or new
test cohort. The historical fixed-frame0 q.10 endpoint score at frame199 remains
the primary comparable metric. The new auxiliary estimator uses the same fixed
physical frame0 side groups, normalized boundary strains, and an affine OLS fit
of epsilon_y versus epsilon_x over predeclared frames30..199 inclusive; p is
minus the slope. All170 finite pairs and nondegenerate axial strain are required.

The first smoke jobs4684994/4684995 failed safely before full evaluation because
`cached.mode` resolved the Pandas method instead of the `mode` column in the
endpoint-parity guard. The failed code_v1 jobs/artifacts remain recorded. A
schema-only code_v2 retry changed this to explicit column access without changing
the estimator, data or checkpoints. Jobs4685004 joint and4685005 node-only both
completed exit0; walltime/maxmemoryKB were00:09:10/2775128 and
00:03:41/1824288. Frozen code_v2 manifest SHA256
d4fd2275a2f99e2448867cf1e1951f8bd97a6c6e80ea326d7da83e6b6ae9d594.
Recomputed frame199 endpoint values agree with the cached evaluator to maximum
absolute error6.21e-9, confirming identical cohort/inference conventions.

Frames30..199 linear-fit p-ratio R2, AE/rollout respectively, all80/80valid:
joint node_mixed .597394/.952558, joint stiffness_mixed .181996/.634661, and
node-only node_mixed -.125954/.944068. Relative to the unchanged endpoint
rollouts .883940/.566453/.873666, trajectory fitting improves the three rollout
scores to .952558/.634661/.944068. The separately implemented all-frame two-OLS
time-slope ratio agrees closely for rollout: .950525 joint node, .642273 joint
stiffness and .945605 node-only node, all80/80valid. Thus regression materially
reduces mixed-T rollout response noise, especially for node transfer, but does
not rescue AE response: the fitted AE scores are lower than their endpoint
scores. Ground-truth median within-trajectory fit R2 is only .701590 node and
.522299 stiffness, while autonomous predicted curves are much smoother
(.997447/.993628 median for joint); report these diagnostics so smoothing is not
mistaken for faithful reproduction of thermal fluctuations. Single seed123 and
the previously inspected heldout-development IDs40..119 still preclude a seedSD
or untouched-final-test claim. Exact rows, summaries, parity receipts, job
identities and hashes are in `notebooks/results/mini_temperature1000_linear_fit/`.

### Mixed-T-only 1,000-step training pilot — 2026-09-14

User requested testing whether direct mixed-temperature fitting succeeds on the
new 1,000-MD-step data. One preliminary seed123 arm changes only the fitting
domain relative to the corrected joint low-T baseline: AE and propagator train
on `node_mixed` plus `stiffness_mixed` IDs0..19 and select on mixed IDs20..39.
It retains all200 frames, exact unit post-warmup frame weights, the same 2D MP
AE, one-step delta propagator, optimizer/early-stopping budgets, and exclusively
coordinate/latent validation selection. P-ratio and strain remain absent from
inputs, losses and selection. Held-out-development IDs40..119 are evaluated
source-wise on all four sources; primary mixed-T reporting separates AE
reconstruction from autonomous rollout at the historical frame199 endpoint and
the auxiliary normalized boundary-strain affine fit over frames30..199.

Frozen snapshot `mini_temperature1000_mixed_only/code_v2`, code-manifest SHA256
`a3a64c53127ac5b41e350f44c9b1a0edf44b6e06de6c201793478ea35a28de28`,
data-manifest SHA256
`2b33efffe3e690d9d4dacfd66c22de28b97ad4848aba9e1a05229a41d8198c51`.
Fit job4685141 is running; collector4685142 is held `afterok` and will perform
all post-fit evaluation. Both request `mendels_q`,8CPU,16GB,
`place=free:shared`; no host pin. Static recipe assertions passed before
submission. An earlier code_v1 snapshot was never submitted because sandboxed
`qsub` could not create its temporary script; it is retained in the ledger and
is not a job or result. Exact recipe, dependencies and status are in
`notebooks/results/mini_temperature1000_mixed_only/job_ledger.json`. Results are
pending; one seed means no SD or final-test claim.

Mixed-only fit4685141 completed exit0 after02:27:48 using3500196KB; collector
4685142 is still running, so auxiliary fitted results and final completion
receipt remain pending. Its completed held-out frame199 endpoint files already
give AE/rollout p-ratio R2, respectively: node_mixed .928575/.905186 and
stiffness_mixed .798073/.444567, all80/80valid. Against the matched low-T-trained
joint model (.745768/.883940 node and .482135/.566453 stiffness), direct mixed-T
fitting improves both AE reconstruction scores and slightly improves node
rollout, but reduces stiffness rollout. These are endpoint-only partial
collection results; do not substitute them for the pending frames30..199 fit.

Post-fit mixed-only prefix evaluation was requested for notebook02. It freezes
the completed seed123 `joint_mixed_only` AE/propagator and evaluates the same
heldout-development IDs40..119 for node_mixed and stiffness_mixed. At every
horizon37..199 it computes normalized boundary-strain OLS over
frames30..horizon and reports AE and autonomous p-ratio R2 with valid/total;
horizon199 must reproduce the completed linear-fit artifact and cached endpoint
values. No training, selection, or checkpoint mutation occurs. Frozen snapshot
`mini_temperature1000_mixed_only/prefix_linear_fit/code_v1`, code-manifest
SHA256 `a5f3c9de79bd1b1fc2d2058640fa71b5f2a8cb538f611797368156f46ab97d6d`;
job4687548 uses mendels_q,8CPU,16GB,placefree:shared without a host pin.
It completed exit0 after00:09:51. Horizon199 reproduced the prior linear values
to <1.2e-16 and endpoint caches to <6.3e-9; checkpoint hashes were unchanged.
Autonomous fitted p-ratio R2/MAE is .919994/.083793 node_mixed and
.460448/.142120 stiffness_mixed, both80/80 valid.

Notebook01 was subsequently narrowed to mixed-T heldout statistics at user
request. Latent-plane, selected-trajectory, response-range and initial-latent
correlation panels now show only node_mixed and stiffness_mixed; the four
training trajectories remain solely as an exposure audit. Final p-ratio is the
same normalized boundary-strain OLS over frames30..199. Heldout Pearson/
Spearman correlations are node z0 -.878/-.874, node z1 .882/.876, stiffness z0
-.764/-.767 and stiffness z1 .754/.759, all80/80. The selected raw trajectory
coordinate is z0 based on the stiffness-mixed heldout correlation, and its
initial inset is restricted to frames0..5 (MD steps0..5000). Notebook
executes17cells,4inline PNGs,0errors. Builder/notebook SHA256 are
`631a866d787cb35314a9db747ff5019757c64960c8faec892ba813db67576efa`
and `f78624c6f567916b05521d557caed89b640c8ce8df42b76eb1d67afbb5a8f631`.

Notebook02 and notebook03 were then converted to mixed-T-only paper-facing
views. Notebook02 retains the honest note that the frozen 1/20 comparison
models trained on paired all-four data, but every displayed trajectory,
correlation, richness statistic and response panel is node_mixed or
stiffness_mixed. It adds the separately trained mixed-T-only model's prefix-R2
curve and final predicted-versus-true fitted-p-ratio scatter. It executes
16cells,5inline PNGs,0errors; builder/notebook SHA256 are
`9f6d94c36921785bdf35a13831b355cc230bd12cb60984ad916d478748c8bdd6`
and `278cdc88f8dbe083c269b32e55b9e0cd1757d8355734f4dd4aad6baaa12a5879`.

Notebook03 now uses stiffness_mixed IDs40..119 and strain-OLS p-ratio for the
entire 1-versus-20 analysis. Within-trajectory and matched-network panels are
restricted to frames0..5. Low-T-only frozen-input and count-morph panels were
removed rather than relabelled. Static reference descriptors are reused only
through exact paired network IDs, while latent states and responses are mixed-T.
Feature and edge gradients were recomputed response-blind on stiffness_mixed
ID40 from the frozen one/20 encoders in
`edge_attribution_one_vs_twenty_mixed`; exporter SHA256
`9d740a1b2ecbbfc3516b1a15ab9ab1ce4cd3a091865c453041ed0d6d6bd3fad6`.
Notebook03 executes15cells,6inline PNGs,0errors; builder/notebook SHA256 are
`138a364631b99fdd6f4d440b01a5d8baedb582791abd98eb5b05daaaab745c2e`
and `92691dd2afc0b3c4fc0e1ff9af35395b967db7f36cdc8ffebddd09ddfc2ac060`.
Notebook06 now contains an executed inline two-panel endpoint-R2-versus-MD-step
comparison (solid mixed-trained, dashed low-T-trained) for AE and autonomous
rollout; no figure file was exported.

### All-four 1-versus-20 trajectory coverage study — 2026-09-14

User requested rebuilding notebooks01/02 on the four new 1,000-MD-step mini
sources, first with one training trajectory per source and then20 per source to
test whether held-out latent geometry becomes richer. Two matched seed123 arms
use `node_low`, `node_mixed`, `stiffness_low`, and `stiffness_mixed`: `all4_one_each`
trains exact ID0/source, while `all4_twenty_each` trains IDs0..19/source. Both
validate on IDs20..39/source and reserve the previously inspected development
cohort IDs40..119/source for frozen evaluation. They retain the same fresh2D MP
AE, all200 frames, exact unit post-warmup frame/source weights,60epoch/patience12
budgets and frozen one-step delta propagator used by the corrected 1,000-step
study. Coordinate MSE and latent-delta validation losses alone select
checkpoints; p-ratio/strain are post-fit only.

The comparison is not solely a trajectory-count intervention: low/mixed same-ID
records are paired versions of the same topology. The one-each arm therefore
has4 trajectories but2 unique structures; twenty-each has80 trajectories but40
unique structures, changing both structure coverage and thermal-trajectory
coverage. Exact source/ID/temperature exposure will be saved per arm. Notebook
interpretation will quantify scale/rotation-aware latent geometry, rather than
claiming richness from arbitrary raw-axis orientation alone. Frozen evaluation
will separately report AE reconstruction and autonomous rollout source-wise at
the historical frame199 endpoint, plus the auxiliary frames30..199 normalized
boundary-strain affine fit for mixed sources only.

Frozen snapshot `all4_coverage_1000/code_v2`, code-manifest SHA256
`aefc3fe6b50956f06d5e8c9759984e2818f47f224f64e16b7a701b515165cfcb`,
data-manifest SHA256
`c92162d0c2fd475e41f13f41596685274b346e7d81d23e9115aba317948c824a`.
Fit jobs4685192(one-each) and4685193(twenty-each) are running; collectors
4685194/4685195 are respectively held `afterok`. All use `mendels_q`,8CPU,16GB,
`place=free:shared`, no host pins. Static parity/configuration, auxiliary-metric,
syntax and whitespace checks passed. An initial sandboxed submission failed
before PBS accepted any job because it could not create a `/var/tmp` qsub file;
the unsubmitted code_v1 snapshot and failure are preserved. Exact dependencies,
hashes and status are in `notebooks/results/all4_coverage_1000/job_ledger.json`.
Results and notebook rewrites are pending; this is one seed and not final-test
evidence.

Initial code_v2 jobs did not yield models. One-each fit4685192 failed after its
smoke warmup and twenty-each fit4685193 was cancelled before the same inevitable
assertion; dependent collectors4685194/4685195 became terminal without running.
Cause was provenance-only: exposure export zipped a flat source-interleaved
simulation list against one source-local ID range, so its newly added coverage
assertion failed. No post-fit evaluation or production result was collected.
Code_v3 enumerates each source separately and asserts every exact
`(split,source,ID)` once; no data, architecture, objective, optimizer, split or
selection rule changed. Corrected fits4685209(one-each) and4685210(twenty-each)
are running; collectors4685211/4685212 are held afterok. Code-v3 manifest SHA256
`5ae03a4cdc5685149e907515e72db3e32468922a72e79a593389505059eca501`;
data manifest remains unchanged. Failure and recovery are preserved in the same
job ledger.

User then reduced validation to exactly30 trajectories for speed and requested
a matched restart. Code_v5 assigns node_low/node_mixed IDs20..27 (8 each) and
stiffness_low/stiffness_mixed IDs20..26 (7 each):30 trajectories representing15
paired unique structures. Held-out evaluation remains IDs40..119/source. Active
v3 one-each4685209 and v4 twenty-each4685237 plus held collectors4685211/
4685238 were cancelled and preserved; neither supplies a result. Matched code_v5
fits4685259/4685260 are running with collectors4685261/4685262 held afterok.
Code-manifest SHA256
`7555cdce89d21d542eb492f03f5be37b58941e323c6ea3093e3b7b926f99259a`;
data hash and all non-validation settings are unchanged. This is a deliberate
selection-cohort change, so code_v5 supersedes earlier attempts for the notebook
comparison.

Code_v5 fits completed exit0: one-each4685259 in01:33:42 using3437184KB and
twenty-each4685260 in04:00:38 using4154220KB. Primary collectors wrote complete
heldout endpoint/per-step/per-network/latent artifacts for all four sources and
320 trajectories per arm, then exited1 only in the auxiliary launcher because
the submitted tag already began with `_` and `linear_run.py` added a second
underscore. The empty double-underscore directories and failed collector job
identities4685261/4685262 are preserved. No primary artifact or checkpoint was
overwritten. Corrected post-fit-only jobs use the identical frozen evaluator
with tag `retry_v5_validation30`: one-each4685853 completed exit0 in00:08:04,
3223052KB; twenty-each4685855 completed exit0 as recorded below. Repair PBS SHA256
`15c1a80d866fcfb3422edf3d53e06e028867e62cc628b736b71665b4c1758cdf`.

Notebook01 is now an executed cached-results narrative for the one-each model.
User clarified its primary objective is heldout correlation of initial latent
coordinates with final p-ratio, not successful autonomous rollout. Using
source-appropriate post-fit truth (low-T physical endpoint frame199; mixed-T
normalized boundary-strain OLS frames30..199), Pearson(initial z0, final p) is
-.873 node-low, -.878 node-mixed, -.858 stiffness-low and -.764
stiffness-mixed; z1 is +.877,+.882,+.848,+.754. Spearman z0 is
-.864,-.874,-.789,-.767. Every entry is80/80valid. These are descriptive
heldout correlations from one fixed2D seed, not supervised probes or model
selection. The secondary mixed-T position-prediction R2 is low, as expected in
this extreme limit: node AE/rollout .0810/.1311 and stiffness .0118/-.0317,
all80/80valid. Do not use rollout failure to negate the strong initial-latent
ordering claim.

The new mini stiffness-optimized sources derive from
`data/mini_stiff_optimized_200_1500.pt`; they are distinct from the historical
`data/reid_200_frames.pt` collection even though Reid is also described as a
stiffness-optimized family. Notebook labels preserve `stiffness optimized` and
state this distinction explicitly. Notebook01 contains the requested coloured
heldout z0 trajectories with per-source initial zooms, source-wise initial-z
correlation panels/tables, and a secondary AE/rollout section. Current execution
has 21 cells, 6 inline PNG outputs and 0 errors; no figure files were exported.

Notebook01 now also visualizes a frozen low-T near-collision: held-out node ID47
and stiffness ID106 at frame33 (MD step33,000), under the same one-each seed123
AE/job4685259. Raw 2D latent separation is0.000173, yet raw box ΔLy is
+0.014122 versus−0.011566; both have box x-strain−0.003300 (2/2 valid).
The inline three-panel plot shows z0 crossing, raw latent paths and box ΔLy;
the exact values are printed beneath it. This is descriptive heldout-development
analysis, not a new fit or final-test result. Source paths/hashes and the
framewise box export are in
`notebooks/results/all4_coverage_1000/notebook01_crossing_pair/`.

Follow-up interpretation audit (same frozen checkpoint, no training): at frame33,
node47 has z0=1.471606+0.851036 and stiffness106 has z0=0.777404+1.545375.
Their nearly coincident codes result from different starting offsets and travel.
Native AE boundary p-ratio is−0.120635/−0.101643 versus truth
+0.336175/−0.229816 (node/stiffness, 2/2 valid). A frozen decoder probe supplying
the identical mean z=(2.32271099,−3.61238480) to each graph's reference context
still gives−0.120635/−0.101644: graph conditioning does not recover the opposite
transverse signs here. These are single-frame boundary diagnostics, not fitted
p-ratio or rollout scores. The pair's normalized displacement RMS nearly agrees
at33, but cohort analysis of3,237 cross-family z0 crossings over frames1–99
finds median relative RMS mismatch7.27%, versus5.62% for arbitrary same-frame
pairs. Thus the near-collision is not evidence of a common physical response.

Standalone visualization: `notebooks/dynamics/10_latent_crossing_and_response.ipynb`
(9 cells, 3 inline figures, 0 execution errors). Same frozen pair/recipe above;
shows raw z0/z1 timing, initial-plus-movement decomposition, raw box ΔLy, and
matched boundary-estimator truth versus AE p-ratio (frame33 valid2/2). No
training, rollout evaluation, or figure exports. Built by
`tools/analysis/build_latent_crossing_notebook.py`.

Notebook10 now includes the frozen twenty-each comparison (17 cells, 6 inline
figures, 0 errors). Seed123; fit4685260, corrected evaluator4685855; train IDs0–19
per source, held-out-development IDs40–119. The original node47/stiffness106
pair is separated at frame33 (raw distance1.365203); true/AE p-ratios are
+0.336175/+0.225485 and−0.229816/−0.109867 (2/2 valid).
Searching6,400 cross-family pairs over saved frames1–99 without response-based
selection finds node102/stiffness51 nearest at frame2 (distance0.000750).
There are2,525 strict synchronous interpolated z0 crossings; minimum remaining
z1 gap0.000072, not an exact2D collision. Selected frame2 true/AE p values:
node+0.208412/+0.071937, stiffness+0.054242/+0.146651; frame199:
node+0.141390/+0.137291, stiffness−0.352315/−0.362360 (all2/2).
Whole-cohort endpoint AE p-ratio R² at frames2/33/99/199 is
node−11.7526/.9618/.9847/.9823, stiffness−3.1681/.7153/.9738/.9430
(80/80 each; one seed, no seed SD). Early ratios have small compression
denominators; these are endpoint diagnostics, not thermal fits or rollouts.
Audit tables and input/code hashes: `notebooks/results/all4_coverage_1000/notebook10_twenty_crossings/`.
No new training, final-test use or figure exports.

### More training networks: AE curve pilot — 2026-09-23

Question: does more structure coverage improve notebook10's low-T AE response
curves? Submitted through normal Zeus from zeus-post:4698519, confirmed running
on mendels_q (8CPU/16GB, free:shared). Same frozen v5 2D AE/objective/seed123;
32/source, IDs0–19+28–39; unchanged validation node20–27/stiffness20–26 and
held-out-development40–119. Four training sources; low-T evaluation only,
frames1–199, pair curve MAE over1–99 plus cohort endpoint p-ratio R². No
propagator, notebook changes, response-based selection or final-test use.
Artifacts/recipe/hashes: `notebooks/results/all4_more_networks_pilot/` and
`hpc/experiments/all4_more_networks_pilot/`. Results pending; no improvement
claim yet. Collector mode-filter fix was applied immediately after submission;
if the running process loaded the pre-fix script, recover comparison tables
from its saved per-source outputs without repeating training.

Twenty-each corrected post-fit job4685855 completed exit0 in00:08:38 using
3418076KB. Mixed-T normalized-strain-OLS p-ratio R2 is .9380/.9585 for node
AE/rollout and .6399/.5266 for stiffness AE/rollout, all80/80valid. Low-T
frame199 endpoint R2 is .982/.970 node and .943/.630 stiffness, likewise
80/80. These secondary prediction results remain one seed.

Notebook02 has been rebuilt and executed as the matched one-versus-twenty
representation comparison:12cells,3inline PNG outputs,0errors. Response-free
initial-latent covariance effective dimension rises from1.000 in every
one-each source to1.557/1.576 node low/mixed and1.405/1.372 stiffness
low/mixed. Thus the 20-each heldout geometry is materially less line-like.
However, richer2D geometry does not mean stronger single-axis response
ordering. In the 20-each model, Pearson(initial z0,z1) is (.553,.799) node-low,
(.523,.814) node-mixed,(-.478,-.055) stiffness-low and(-.329,.026)
stiffness-mixed. An unsupervised response-label-free PC1 construction gives
absolute Pearson correlation .815/.817 node low/mixed but only .423/.281
stiffness low/mixed, versus .876/.881/.850/.757 for one-each. PC1 uses no
p-ratio to choose its direction; sign is arbitrary and reported in absolute
value. Therefore twenty-each clearly enriches dimensional occupancy and greatly
improves position-derived response reconstruction, while the exceptionally
clean initial-latent scalar ordering of notebook01 is partly lost for the
stiffness branch. Do not collapse these distinct outcomes into one notion of
"better latent space."

### Tiny edge-MLP AE pilot — 2026-09-23 (running)

Question: does initial-z/p-ratio ordering survive a transparent tiny encoder?
Shared low-T-only AE: node/stiffness ID0, all200 frames at1,000MD cadence;
validation node20–27/stiffness20–26, development40–119/source. Seeds123/456/786.
Encoder: standardized reference edge(dx,dy,length,k) and evolving edge changes,
8→8(tanh)→2, mean over both edge orientations (90parameters). Static-node
decoder8→16(tanh)→2; full AE268parameters. No attention/message passing.
Architecture intentionally loses connectivity information in graph encoding.
Adam1e-3, wd1e-5, graph-equal standardized-displacement MSE, batch32,
max300epochs/patience30; checkpoint on validation displacement only.
Compare both initial raw coordinates before/after fitting against frame199
physical endpoint p-ratio (Pearson/Spearman, source-wise counts/seed meanSD),
and separately report position-derived AE p-ratio R² at declared horizons.
No propagator, response supervision, final-test use or notebook changes.
Model: `src/lss/dynamics/tiny_edge_ae.py`; runner/results: `tiny_edge_mlp`.
Small tensor checks passed:268parameters, edge-reversal/node-permutation
invariance and backpropagation. Initial audit below motivates the untrained
control; high correlation alone is not evidence it was learned.
Submitted through normal Zeus: seeds123/456/786 → jobs4698537/4698538/4698539;
all confirmed running in mendels_q,8CPU/16GB/free:shared. Frozen `code_v1`
manifest SHA256 `6e35f6e6390e76aa5cd7769767b0717e38ccb98434a4765dfb849ccbaa0e16d4`.
Results pending. Exact encoder/decoder matrices and fitted train-only
normalizers will be exported with each checkpoint for an explicit static map.

### One-each initial-latent origin audit — 2026-09-14

User asks whether the strong heldout initial-z/final-p ordering is learned from
the deformation frames or inherited from static graph representation and
dataset structure. Post-fit-only audit4685877 completed exit0 in00:02:23 using
3210188KB. Same frozen one-each seed123 AE, low-T heldout-development IDs40..119,
physical endpoint frame199 truth; no checkpoint, fitting, selection or final-test
change. At frame0 the normalized displacement and evolving-edge-change inputs
are constant; network-varying encoder inputs are reference node positions,
connectivity and reference edge vector/length/stiffness.

Fifteen response-blind random initializations of the identical architecture
with the fitted input normalizers give PC1 absolute Pearson mean±SD .802±.292
node and .592±.253 stiffness; maxima .982/.899 exceed the trained .876/.850.
Thus architecture plus static dataset structure can expose the ordering without
learning, although one trained seed cannot quantify how much fitting stabilizes
it. Simple static descriptor correlations identify the branch-specific signal:
node mean/std edge length |r|=.985/.974; stiffness std/mean stiffness
|r|=.941/.915, all80/80. Response-blind PCA of all17 standardized descriptors
is not itself strongly correlated (.132 node/.384 stiffness), so the signal is
specific rather than generic maximum variance.

Trained-encoder input ablations support the same attribution. Holding static
edge geometry at its fitted mean reduces node PC1/p |r| .876→.474, while
holding stiffness at its mean retains .941. For stiffness, holding stiffness at
its mean reduces .850→.276, while holding edge geometry retains .854. Setting
all reference-edge channels to their fitted means reduces correlations to
.020/.267. Position centering has small effect; zero-position is explicitly
OOD and not causal evidence. Local normalized-input gradients and every
predeclared descriptor/ablation row are retained. Bottom line: training
compresses/organizes a strong pre-existing optimized-family signal; node
ordering primarily enters through edge geometry/length and stiffness ordering
through the stiffness distribution. Correlated optimized inputs prevent a
unique physical-causality claim. Artifacts and exact hashes:
`notebooks/results/initial_latent_origin/`; runner/PBS:
`hpc/experiments/initial_latent_origin/`.

### Nested latent-count morphology study — 2026-09-14

User requests watching the2D latent plane morph with training coverage. Frozen
matrix counts1/2/4/8/10/12/14/20 per source × seeds123/456/786; nested train
IDs0..count-1, identical code-v5 four sources/all200frames/30-trajectory
validation, ordinary warmup plus uniform coordinate-only AE fitting. No
propagator training is needed for this representation question. Existing
seed123 count1/count20 checkpoints are reused without retraining; all other
cells train fresh. Heldout low-T IDs40..119 give initial latents plus eight
response-blind fixed-ID full trajectories/source. Collector alignment is
centered Frobenius-normalized orthogonal Procrustes on identical heldout latent
coordinates without p-ratio. Effective dimension, PC1 variance and trajectory
geometry are primary; low-T endpoint p is post-fit correlation only and never
selection. Matrix has24 cells. Gate job4685881 runs the reused count1 seed123
cell before release of remaining cells. An attempted single-element PBS-array
syntax was rejected before job acceptance; no experiment resulted. Runner SHA
`f0746dc2da6e315a0dc9e99d93f9833acaf754be7e7571d39d489ce23eb332e0`,
matrix SHA`7e757a706d2efd9b474203b9a4faa252d1d977efe46a53c2a54e897f48fe30a2`,
PBS SHA`f082d230923161874fc35869abfd137d14c7f1ac8c8e3b34e15edb1b491d0650`.
Shared mendels_q8CPU16GB/placefree:shared, no host pins. Results pending.

Gate4685881 completed exit0 in00:02:17 using3201468KB and reproduced the reused
count1/seed123 checkpoint hash exactly while writing initial and fixed-ID
trajectory latents. Remaining23 cells submitted as PBS array4685887[1-23]; all
23 entered running state. The earlier rejected `-J 0-0` command remains a
submission-syntax failure only, with no accepted job. No collector is submitted
yet; aggregation and notebook morph plots wait for all declared cells and must
report failures rather than silently dropping them.

Notebook02 has been re-executed with four source-specific AE versus autonomous
rollout endpoint-p-ratio R2 curves across every stored MD horizon. Mixed-T
curves are explicitly labelled as thermally noisy frame-specific endpoint
estimates; the less-noisy final frames30..199 strain-OLS values remain in the
table. Notebook02 now has4inline PNG outputs and0errors.

Historical notebook03 was replaced at user request and executed as the exact
one-versus-twenty comparison. At the user's correction, every coordinate-level
analysis now uses the model's actual two learned variables z0 and z1, without a
PCA replacement. It reports initial z0/z1, final delta-z0/delta-z1,
full-trajectory z0/frame and z1/frame slopes, arc length, straightness,
effective dimension, pooled motion/frame correlations and all source-wise
correlations with source-appropriate final p. The effective-dimension scalar is
computed from the two-coordinate covariance but does not rotate the displayed
variables. It also embeds the completed direct-z static-input/random-encoder
attribution audit. Thirteen cells,3inline PNG outputs,0errors; no figure files.

Direct-z audit job4685893 completed exit0 in00:03:25 using3224924KB. For the
trained one-each encoder, initial z0/z1 absolute Pearson correlations with
low-T endpoint p are .873/.877 node and .858/.848 stiffness, all80/80. Across
15 response-blind random initializations of the same architecture, mean raw
z0/z1 absolute correlations are .766/.693 node and .587/.531 stiffness; maxima
are .983/.970 and .895/.897. Trained direct-coordinate ablations give node
baseline .873/.877, geometry-held-mean .473/.474, stiffness-held-mean
.940/.941 and all-reference-edges-held-mean .020/.020; stiffness gives
.858/.848, .862/.852, .267/.279 and .255/.270 respectively. This preserves the
earlier conclusion in the actual learned coordinates: node ordering primarily
enters through static edge geometry and stiffness ordering through the static
stiffness distribution. The earlier PC audit is retained as auxiliary history
but is superseded for notebook03 presentation by
`notebooks/results/initial_latent_origin_z/`. Runner SHA
`d6a331d4a6400da4608a84b8d1831da5e35812a05c7108ea49fe106eccea2e12`.

### Direct-z structure, motion and edge-attribution narrative — 2026-09-14

Notebook03 was expanded and re-executed:19cells,7inline PNG outputs,0errors.
All coordinate plots use raw learned z0/z1. For low-T heldout IDs40..119,
one-each corr(initial z0,z1) is -.9996 node and-.9994 stiffness, so squared
shared variation is.9993/.9988 and effective dimension1.000/1.000. Twenty-each
corr is+.4939/.5289, squared shared variation.2439/.2798 and effective
dimension1.557/1.405, all80/80. Thus the nominally2D one-each state is almost
one duplicated coordinate, while twenty-each represents independent initial
graph variation. This does not claim every within-network path is intrinsically
2D: at frame199 twenty-each corr(delta z0,delta z1) is.963 node and.797
stiffness. Response appears strongly in motion: twenty-each endpoint absolute
corr(delta z0/z1,final p) is.977/.982 node and.886/.978 stiffness, all80/80.
Framewise motion-response curves including frames10/50/100/199 are inline.

The encoder receives no source ID, temperature or p-ratio. Node/stiffness
families separate only through positions and compact reference/current edge
direction, length and stiffness features. Raw initial family centroid distance
divided by pooled within-family RMS is6.351 one-each and1.403 twenty-each. The
large one-each offset is therefore not a supplied label. Twenty-each static
associations suggest node z1 retains mean-edge-length ordering(r=-.795), while
stiffness z0 retains stiffness-spread ordering(r=.454) and stiffness z1 also
represents gyration radius(r=.682). These twenty-each descriptor results are
correlational; the frozen-input ablation remains completed only for one-each.

New response-blind edge audit uses fixed heldout graphID40 for node-low and
stiffness-low and passes the exact graph through frozen one-each and
twenty-each AEs. Per physical spring importance is the L2 norm across z0/z1 of
gradient-times-normalized-input, combining current/reference routes and paired
directed entries; geometry uses vector-x/vector-y/length and stiffness uses the
stored-stiffness channel. It is a local sensitivity visualization, not a causal
edge deletion. Notebook drawing now uses `graph_utils.draw_graph` with
`periodic_edges=False`; the utility removes51/439 node and61/830 stiffness
wraparound edges from the display only, while all remain in encoder attribution.
Across visible edges, one-versus-twenty importance-rank correlation is-.181 for
node geometry and+.170 for stiffness-channel influence; overlap between each
model's top10% visible edges is.077/.039. This demonstrates strong
fitted-representation reorganization on the same graph but not a unique causal
spring set. Artifacts:
`notebooks/results/edge_attribution_one_vs_twenty/`; script SHA
`97cd11c246ecc4aa1e77e3c17ed4b1b5d43087c385e21743a292e69f90ef3402`,
CSV SHA`aee26e7799047b4f69c797a83ef0db7397b380930684a74513fff0a25801777b`,
same frozen AE hashes recorded in metadata.

Notebook02 was rebuilt/re-executed:14cells,4inline PNG outputs,0errors. Its
latent-plane figures now omit noisy mixed-T paths and overlay all80 node-low and
80 stiffness-low complete200-frame trajectories on common z0/z1 axes for the
twenty-each model. Mixed-T remains in source-wise response evaluation. Every
displayed p-ratio-R2 horizon curve is clipped at zero from below with ylim
0..1.03; exact negative values remain unchanged in displayed tables and source
CSVs. Notebook02 builder SHA
`032201e421914dbec2f05adfe3cdf9352a23c7af5c593232fdfad4e71c0cd918`.

Latent-count scheduler audit corrected a misleading parent-only qstat result:
original array4685887 still had tasks present. Nine/24 cells were complete
(indices0,1,2,7,8,10,16,17,18). Fifteen incomplete cells are exactly
3-6,9,11-15,19-23. Their original partial artifacts were preserved under
`latent_count_morph/partial_4685887/`. A first rerun overlapped briefly while
PBS applied old-task cancellation gradually; all involved processes were
stopped and those outputs preserved separately under
`latent_count_morph/overlap_window_4685947_4685950/`, so none can enter the
collector. Final clean reruns began only after no relevant old/brief-rerun task
remained active: jobs4685958[3-6],4685957(index9),4685956[11-15] and
4685959[19-23], all15/15 verified running. Existing nine completed cells remain
untouched. Final collection requires completed.json status/hash/count/seed
validation and will present per-seed raw z0/z1, z0/z1 redundancy and covariance
effective dimension; no PC coordinate will replace the two learned variables.

### Early latent turn and completed count morphology — 2026-09-15

User notes endpoint directionality can hide nonlinear motion near the beginning.
Notebook03 now measures, for each heldout low-T trajectory, the angle between
z(frame)-z(0) and its own eventual z(199)-z(0), using raw z0/z1 only. All80/80
trajectories are valid at every frame. One-each is already straight at frame1:
median angle.823degrees node and1.089degrees stiffness; at frame10
.863/1.072degrees. Twenty-each has a reproducible initial turn in seed123:
frame1 median30.994degrees node and22.073degrees stiffness, frame10
11.296/9.468degrees, frame20 6.334/7.547degrees. Frame20 median perpendicular
deviation remains only.016/.022 of final displacement, so this is an early
turning transient rather than a large loop. Notebook shows median plus10–90%
bands through frame50 and centered initial paths for fixed IDs40,54,68,82,96,
110 with frame10/20 markers.

All24/24 latent-count matrix cells are now complete and passed strict
completed.json status/count/seed/runner/data-hash validation. Final clean jobs:
4685958[3-6],4685957(index9),4685956[11-15],4685959[19-23]. Collector outputs
are `notebooks/results/latent_count_morph/collected/`; collector SHA
`bfaffd69398f31bb0e138ec45a45e34aba31c03d9cae96c57f151dc59b085421`,
raw-z metrics CSV SHA
`619a319f5c0e227c956f47c9b9bdd653ea7c41a8e09f1d88a95f724bf57679e9`,
aggregate SHA
`3451f84ed5279718ac145ac2694c2f74f843db98efd78729881eb81f1d17720d`.
Each count/source has3seeds and240/240 initial heldout states; early motion uses
24/24 fixed response-blind trajectories(count/source aggregated across seeds).

The count transition is not monotonic. Seed-mean squared z0/z1 shared initial
variation for node is.954,.519,.995,.200,.196,.330,.229,.272 at counts
1,2,4,8,10,12,14,20; stiffness is.997,.492,.922,.352,.201,.075,.298,.191.
Mean effective dimension at count20 is1.536±.266 node and1.485±.170 stiffness,
versus count1 1.012±.016/1.001±.002. Count4 often collapses back to a line;
counts8–20 generally use more of the plane but retain seed variability. Hence
broader exposure makes a richer2D solution available but does not enforce a
smooth or unique coordinate decomposition. Notebook03 now contains seed123
raw-z small multiples for all eight counts, multi-seed mean/SD curves, the
early-turn analysis and the actual-graph edge maps. It executes23cells,
11inline PNG outputs,0errors; builder SHA
`adb48ec0fcdd44842f6a20e0a94b7c33f630009a761e7a3a66044cc730e30b0b`.

### Notebook03 stiffness-low focus — 2026-09-15

Notebook03 was reduced to the stiffness-optimized low-temperature source for
the one-versus-20 mechanism analysis. Mixed-temperature and node-optimized
rows are absent from its correlation dataframe, displayed tables and plots;
this is a presentation/analysis-scope change and does not retrain or select a
model. The notebook retains raw z0/z1 initial structure, frozen-input static
attribution, the complete count-morph sequence, early-turn analysis,
framewise motion/p-ratio correlations and the graph-utils stiffness-edge map,
all for stiffness-low only. It executes21cells,10inline PNG outputs,0errors;
builder SHA
`8add638355d7da510af0abae9b5a6be6a6fb010f9c774b02813681c153ce2158`.

### Notebook03 matched early trajectory — 2026-09-15

Notebook03 now ends with a response-blind matched visualization for heldout
stiffness-low networkID40. The identical physical200-frame trajectory is
passed through the frozen one-each and twenty-each encoders. Full raw z0/z1
paths, frames0–20 zooms, eventual latent direction, physical transverse-versus-
axial boundary-length change and per-frame latent/box increments are shown
together. Boundary lengths use fixed frame-0 side-node sets selected by
`graph_utils.directional_side_indices_from_box` at quantile.10:
Lx=mean(right x)-mean(left x), Ly=mean(top y)-mean(bottom y). Stored
x/y-deformation columns are cumulative delta Lx/delta Ly, not normalized
strains; p=-delta Ly/delta Lx for this cohort. The one-each
angle to its eventual latent direction is1.078,1.074,1.020degrees at frames
1,10,20; twenty-each is22.701,9.809,7.574degrees. The identical physical
p-ratios are+.159,-.035,-.061 at those frames. For network40 frames1–20,
twenty-each corr(delta z0,-delta Ly/delta Lx)=.9659 and
corr(delta z1,-delta Ly/delta Lx)=.9909, versus-.8459/.8389 one-each;
corr(delta z0,delta z1) is.9232 twenty-each versus-.9999 one-each. Mean
per-frame delta Lx is-.0016609 with SD.00000596, confirming essentially
prescribed axial increments, while delta Ly SD is.0000970 around mean
-.0001016. Thus the curve resolves the changing transverse response under
controlled axial compression; z still retains internal displacement
information because the AE reconstructs every node. Notebook
executes23cells,11inline PNG outputs,0errors; builder SHA
`ac90e03684e53e4b7fa1c869de9235d91dea4b1474fd5505a84865555db84c66`.

### Notebook03 correlation-scope separation — 2026-09-15

The former single feature/final-p heatmap conflated statistical units and was
replaced by three explicitly distinct views, stiffness-low only. Global static
correlations are computed once across80 heldout networks between initial z and
frame199 p-ratio; global motion correlations are computed across the same80
networks between endpoint delta-z/full-path slope and frame199 p-ratio.
Within-trajectory correlations are instead computed separately inside each
network over frames1–20. Their primary dynamic comparison is per-frame
delta-z=z(t)-z(t-1) against the simultaneous incremental boundary response
-delta-Ly/delta-Lx; a secondary table retains cumulative
z(t)-z(0) versus frame p-ratio and labels it separately.

All within-trajectory cells are80/80 valid. Median absolute per-frame
correlation for one-each is.786 z0/.779 z1; twenty-each is.956/.986.
Corresponding10–90% intervals are.228–.894/.243–.889 one-each and
.467–.992/.733–.998 twenty-each. Global initial-z/final-p absolute
correlations remain.858/.848 one-each versus.478/.055 twenty-each; global
endpoint delta-z/final-p is.019/.140 versus.886/.978. Notebook executes
23cells,11inline PNG outputs,0errors; builder SHA
`9c95c0b6166c063c7b0d23484aeca4375d6032d76464709a5b827af92f4ce971`.

Notebook03 presentation was subsequently stripped of narrative interpretation
at user request. Markdown now contains section headings only; plots, axis
definitions, code, exact tables and valid/total counts remain. The concluding
takeaway cell and interpretive print statements were removed. The notebook
executes22cells,11inline PNG outputs,0errors; builder SHA
`c222acfd9387f464cf8e5d84163cf23b8323522b683be040142735143d9a497b`.

Notebook03 headings and panel titles were made self-identifying without
restoring narrative prose. The correlation section now names both targets:
final p-ratio for across-network statistics and -delta-Ly/delta-Lx for
within-trajectory framewise statistics. The matched network40 figure now
enlarges frames0–20 directly in centered coordinates normalized by the
frame20 latent-displacement norm. Its one-each panel visibly follows the
frame199 ray(frame1 turn1.1degrees), while the twenty-each panel visibly
curves away from that ray(frame1 turn22.7degrees); cumulative boundary-length
change and standardized framewise response remain below. Notebook executes
22cells,11inline PNG outputs,0errors; builder SHA
`3d3c1aa0cd94b907879974a62c8d603ea8d629ac8ec7cf1f4490b306082096e0`.

Notebook03's final matched-network figure was simplified for supervisor-facing
presentation. It now shows only the raw z0-versus-z1 path over frames0–20 and,
for each frozen encoder, the frame-to-frame z0 and z1 changes beside the true
framewise p-ratio. Frame20 displacement
normalization, the frame199 direction, cumulative boundary-length change,
the colorbar and auxiliary direction/increment tables were removed. This is a
source-only presentation change: the notebook was deliberately not executed,
and its outputs were cleared pending the requested manual rerun. Builder SHA
`84eece346e65afd69b396b8888932fa9ebeba2578d03c6642053d65c335c4cf0`.

Notebook03's endpoint-motion/final-p-ratio analysis was removed because it
mixed an across-network endpoint statistic with the intended within-trajectory
motion question. The remaining correlation figure separates (1) initial z
versus final p-ratio across 80 networks from (2) latent change since frame 0
versus y-deformation within frames 1–20 and (3) latent change since frame 0
versus the same-frame p-ratio. The across-network delta-z curves, endpoint
delta-z/slope bars, all frame-199 direction diagnostics and their extra tables
were removed. The matched example likewise overlays z0/z1 change since frame 0
with the ground-truth frame p-ratio. This was a source-only edit; outputs remain
cleared for manual rerun. Builder SHA
`22450e1da3077dbc6cbad2eab5ef98d5c218eb76c7cfe4d0cd30b3e3d8c68c59`.

Notebook03 now ends with a concise supervisor-facing interpretation of the
learned mapping. It states that a complete graph frame (positions,
connectivity, edge geometry and stiffnesses) is encoded into z0/z1 without an
optimization-class or p-ratio input; one-network training can rely on broad
source identity and dominant deformation, whereas twenty-network training
must also resolve within-source structural and response variation. It also
explains the node- versus stiffness-optimized separation through their
different geometry/edge-length and stiffness-pattern inputs while warning
that latent axes have no fixed physical names. Source was rebuilt and checked
without notebook execution; outputs remain cleared. Builder SHA
`2edc16e2cb60ad7c918eecf929983ca233a7c42231fec30d8901f0af5df8dc10`.

Notebook03 stiffness-edge rendering now compares the completed mini_shared
original-only arms on the identical held-out stiffness-optimized network40 at
frame0: stiffness_only trained on20 stiffness originals(job4682401, AE SHA
78d2eb176be0dbf825a1d743db5db3efb9a00826f7ff84c384d77b4052dc858b)
versus joint_original trained on20 stiffness+20 node originals(job4682402, AE
SHA429dbcf558cfe3b39047c1cd48cb2c502c7c2ab56bbf7dd1ee5677a2433ebb1c).
No mixed-T data, source label or p-ratio enters either fit or this attribution.
For each of830 undirected model edges, the response-blind score is the L2 norm
across z0/z1 of gradient times normalized stiffness input, combining current
and reference edge routes. Panels percentile-rank scores separately; edge
width still shows physical stiffness, and periodic drawing edges are removed
through graph_utils. Rank correlation between the two complete edge score
vectors is.988552 and their top10% overlap is65/83(.7831). This is local
sensitivity, not causal deletion. Artifacts are under
notebooks/results/edge_attribution_stiffness_vs_joint; CSV SHA
509ffb7206cad050c04c81cef8980496247f8a13ba366f919930d107b223642f and
exporter SHA91e6bd743a4a0bb98d929a00dce55d40f5e2a6a3ca4b6035a642e9e99fc1fdb9.
Notebook source was rebuilt and outputs cleared without execution; builder SHA
`30d372d938537a0da93e5c6ee59e603ad24b491c69431f38578a1e7f7d0ddebe`.

Notebook03 feature attribution and full count-morph paths were extended at user
request. The earlier stiffness-vs-joint edge export incorrectly combined
compact current-edge channel3(relative stretch) with reference-edge
channel3(stiffness) under a stiffness label. It is superseded: edge colors now
use reference-stiffness only. A response-blind gradient-times-input audit on
the same heldout stiffness-original network40 ranks eight actual input
families at frames0 and20, L2 aggregated across z0/z1 and displayed as a share
of the sum of family scores. At frame0, stiffness-only influence shares are
reference node positions61.66%, stiffness26.42%, reference edge directions
8.13%, reference edge lengths3.79%; joint-original shares are84.40%,3.41%,
6.31%,5.87%, respectively. At frame20, stiffness-only is reference positions
51.25%, relative stretch13.89%, edge-length change10.49%, stiffness9.22%, node
displacement7.42%; joint-original is reference positions58.86%, node
displacement12.44%, relative stretch8.91%, edge-length change6.31%, reference
edge length5.42%, reference edge direction5.33%, stiffness2.09%. These are
single-graph local sensitivities and family sizes differ; they are not causal
deletions or population estimates. Feature CSV SHA
`1f8e0afe36002648d9ad440a56d77afa804d7fd32d7c0fe53f7039cbc1a4030e`;
exporter SHA
`42e46a1361426401db25653e9438929cf294901730bf3d3d0d3afd219a6b8b1a`.

The previous count-morph cache held only8 fixed paths/source/cell. Frozen seed
123 checkpoints were evaluated post-fit on the same20 heldout stiffness-low
IDs40..59 for all200 frames and counts1,2,4,8,10,12,14,20:32000/32000 finite
latent rows, no training, response input, selection or alignment. First PBS
attempt4687102 exited1 after count1 because intermediate ae_only bundles were
passed to the full experiment loader; no artifact was retained. Corrected
attempt4687109 used the repository ae_only loader and exited0 in15:35,
45:17CPU,3384164KiB peak memory on8CPU/16GB shared. Output SHA
`5a9789408d0d1f3bd22ddaa714c2a65336831ced4a122bb5bd99a66b38017de0`;
exporter SHA
`e3b7e928187e1248ab459e4ad5453a1478cbd628bd6d76004c5341184352135f`.
Notebook03 now plots all20 complete matched paths in every count panel and the
top-five feature-family ranking above corrected stiffness-edge maps. It was
rebuilt but not executed; outputs cleared. Builder SHA
`b0e34f32af5019f14851ae0bace8532cfc982cc9a948579c115baf45cfff1035`.

Notebook03 feature-family gradient-times-input attribution was extended from
frames0/20 to0,1,2,3,4,5,20 for the same heldout stiffness-original network40
and the same frozen stiffness_only/joint_original AEs. The bar small multiples
were replaced with one heatmap per model; cells at or above5% are annotated.
Reference positions remain the largest single family through frame5: their
shares for stiffness-only are61.66,59.94,55.01,52.12,50.60,49.91%, while
reference stiffness falls26.42,25.71,23.02,20.58,18.52,16.82%. Joint-original
reference-position shares are84.40,82.24,76.03,71.82,68.90,66.84%, with
stiffness3.41,3.30,2.96,2.72,2.55,2.43%. Summed dynamic-family shares rise
from0 at frame0 to24.22% stiffness-only and20.30% joint-original by frame5.
Same local/single-graph/non-causal and unequal-family-size qualifications
apply. Feature CSV SHA
`753f4816bf95c6622bc5ac449ab6940d75d65f99fb58189cdebf1af75913f7cc`;
exporter SHA
`bfa5abda923a74e436348d1211523163af0a14d81dcc9fedf065c490b5e5edc0`.
Notebook was rebuilt without execution and outputs cleared; builder SHA
`ae87a9f7e42443294a6bc150d083b449aa2afcfd26debc11e13061b4620cd4f1`.
### Declared 200-MD-step, 500-frame rerun — 2026-09-15

User requested repeating the notebook01-03 mini-temperature studies on the
equivalent MetaForge family stored every200MDsteps, training on exact frames
0..499. The four declared inputs are the node/stiffness x low/mixed branches of
`mini_temperature_200_1500frames`, each200 trajectories x1500 frames. The
predeclared cached prefix ends at MD step99800 and strain.00998; this is denser
but only half the physical horizon of the prior200-frame x1000-step experiment
ending at199000/strain.0199, so results will not be called horizon-matched.

The frozen reproduction scope is: all-four one and20 trajectories/source,
joint low-T, node-low-only, stiffness-low-only, joint mixed-T-only, plus the
nested counts1/2/4/8/10/12/14/20 x seeds123/456/786 AE latent-morphology
matrix. All retain the response-blind fresh2D message-passing AE, exact uniform
post-warmup frame weights, frozen one-step delta propagator, and coordinate or
latent validation selection. Train IDs are nested0..19, validation20..39
(the all-four arms retain their prior30-trajectory validation allocation), and
heldout-development IDs40..119. Low-T primary p-ratio is the physical frame499
endpoint. Mixed-T primary p-ratio is normalized boundary-strain affine OLS over
frames150..499: frame150 is MD step30000, preserving the previous fit window's
physical start. AE reconstruction and autonomous rollout are always separate,
source-wise, with valid/total counts. No response enters training or selection.
The completed cached artifacts will populate separate executed notebooks under
`notebooks/dynamics/200_md_step/`; notebooks01-03 are not overwritten. The
duplicates emphasize complete500-frame/99,800-MD-step paths and early
transients rather than merely transplanting the old plots.

Data preparation streams and verifies every selected per-trajectory SHA rather
than loading the24.1GB of monolithic files together. It checks all240,000
selected frames for finite coordinates, exact time=200xframe, fixed topology,
reference positions and stiffness; it also checks paired low/mixed identities
and no cross-family topology overlap before writing four120-trajectory caches.
`CONTRIBUTING.md` was requested by repository policy but is absent from this
checkout; AGENTS.md and the experiment index/logs were followed. Exact protocol
is `hpc/experiments/mini_temperature200_frames500/protocol.json`. Data audit and
cache job4687224 is accepted on mendels_q with2CPU,16GB and
place=free:shared. Prepare/PBS/protocol SHA256 are respectively
`d2754fec575ff534cc60dcc18f14c3a140f64f102be9fc06cb6604b663638de6`,
`38f2dd1381342c9dac1d6f29db32355326589a6b29b53ceb1387d4609b6c0b6e`,
and `cfa06218edf3683b07b454ac29365e7def7bb270025b6ce30dd328a76255bf8b`.
Training is not released until this job succeeds; no result exists yet.

Preparation job4687224 failed safely after seven seconds on the first graph,
before any cache or model fitting: the auditor incorrectly expected converted
`reference_context_positions` on raw MetaForge graphs. The raw schema has x,
edge_index and four-channel edge_attr; the standard repository loader derives
reference fields from frame0. The schema-only correction audits raw frame0 x
implicitly through finite coordinates and verifies fixed topology plus the
stored last edge-attribute stiffness channel. It does not alter any graph
tensor, split, frame, model or metric. The six fit jobs4687251..4687256 and
collectors4687257..4687262 were terminalized by the failed afterok dependency
and provide no result; they remain recorded for provenance. A corrected data
job and fresh dependent fits will be recorded below.

Corrected data retry4687263 also failed safely at the first graph after ten
seconds, before caches/fits, because PyG reports a present but None `pos`
attribute on this raw schema; coordinates are the tensor `x`. This is another
auditor-only schema error. Retry1 fits4687264..4687269, collectors4687270..
4687275, morph array4687276 and reuse jobs4687277/4687278 were terminalized by
their afterok dependencies and have no result. The next auditor requires a
tensor-valued pos and otherwise uses x[:,0:2]; a four-source 500-frame local
smoke audit precedes another submission.

Data retry2 job4687289 then completed exit0 in00:03:47 with2644884KB reported
memory (Python peak2861980KiB). Four caches contain exact IDs0..119 and frames
0..499; SHA256 node-low/node-mixed/stiffness-low/stiffness-mixed are
`7b011b3d28ad7ffa2ec5ac76f1072265ee8bd502e0aa8aec13ea05c5d6e49f14`,
`6ad055cc9a3b8ef1342d2f6da41c2eda65595d841db694c6dbb81f594ed9f815`,
`6609601352f2577b7001acae9e1a8cd1a94deb9ef61e6ac392fb3e64ca78385c`,
and `38a253595bca5163409d4375ccec2c2010c4c96e734a111981798d2d7fcf40d6`.
All240,000 frames passed the declared checks, paired low/mixed provenance
matched, and standard-loader plus compact-batch smoke produced112x2 nodes,
644 directed message entries and four reference-edge channels.

Full-matrix snapshot code_v4 was submitted only after this audit. Its six base
fits4687313..4687318 and dependent collectors4687319..4687324, morphology
array4687325 plus reuse jobs4687326/4687327, and attribution4687328 were then
cancelled or failed before usable fitting after a packaging-only root error was
found: copied evaluate.py formed `notebooks/notebooks/...` when locating the
frozen historical evaluator. Array tasks15..21 exited1 in9-12seconds; remaining
active work was cancelled rather than knowingly wasting compute. All logs and
job identities are preserved. Root discovery is corrected to locate the parent
containing both src/lss and notebooks; the scientific protocol is unchanged.

Clean packaging retry code_v5 passed static recipe parity and the frozen
evaluator root check. Six fits4687329..4687334 are running; collectors
4687335..4687340 are held afterok. Morphology array4687341 has all22 fresh
cells running; reuse cells4687342/4687343 are held on the matching one-/twenty-
source fits. Four-model attribution4687344 is held on all required fits. Every
job uses mendels_q,8CPU,16GB,placefree:shared without a host pin. Frozen code
manifest SHA256 is
`b0755d5480360df0972c6f115014b0689ddbde65fc3e9fee891c8393a4395825`;
data manifest `347c165087b2698131845318ddc37250a9f6689e5f707c2d7b34f2d8fc073f87`,
protocol `af38299d5496d8782973b3e49315f1b53f1d87727bb14abe76982680ecf471ef`,
and morph matrix
`6cada93be5c090f389efbfd6f8af0b00356e18e5f8bf922122f7cb4300291f7d`.
The morphology output saves all80 heldout initial states and full500-frame
paths for fixed IDs40..59 per low-T source/cell. Results and executed duplicate
notebooks remain pending.

Later status audit found code_v5 base jobs4687329..4687334 exited1 only after
every arm completed its AE and propagator smoke fit. The collector smoke chose
node_low alone, wrote its endpoint evaluation, then grouped an empty mixed-fit
table and raised KeyError(source). Production training never started and the
dependent collectors/attribution did not run. This is an evaluator-smoke bug,
not a model result. All six smoke checkpoints and histories are preserved. The
correction evaluates node_low plus node_mixed during smoke and constructs an
explicitly schematized mixed-fit table. Corrected smoke collectors will gate
production-only fits, avoiding duplicate smoke training.

Morphology is independent of that collector: array cells0..19 completed exit0
with receipts; cells20/21(count20 seeds456/786) remain running. Count1/count20
seed123 reuse jobs were terminalized only because their base-fit dependencies
failed and will be resubmitted after repaired production fits.

### Notebook01 focused initial-state and temperature audit — 2026-09-15

At user request, original notebook01 no longer contains the secondary AE/
autonomous-rollout section; those diagnostics remain notebook02's role. The
trajectory plot now selects the stronger raw initial coordinate using only
heldout stiffness-low correlations. It selects z0, Pearson r=-.858 versus the
slightly weaker z1, and the focused stiffness-low/mixed scatter plots show the
actual training ID0 as a star without including it in the heldout correlation.

The exact source-appropriate position-derived training responses are all
negative: node-low endpoint -.250347, node-mixed frames30..199 strain-OLS
-.258283, stiffness-low endpoint -.028005, and stiffness-mixed strain-OLS
-.027369. Thus the frozen historical model was not trained on a positive
p-ratio example. It was not replaced with a response-selected positive network,
which would violate the response-blind sample-selection rule.

For each mixed-T heldout trajectory, physical transverse fluctuation is the SD
of residual y-displacement after affine removal of its response to x-displacement
over frames30..199. Fluctuation correlates with temperature: Pearson versus
log10(T)/Spearman versus T is .700/.841 node and .754/.901 stiffness, all80/80.
The sole mixed training example is0.1K while heldout data span.1..30K. This
confirms a temperature-dependent fluctuation signal and documents evaluation
outside the training temperature, but alone is not causal proof of temperature
extrapolation because network identity also varies. Notebook executes17cells,
4inline PNGs,0errors. Exporter/builder/notebook SHA256 are
`ad610cbfeb8d6449aa552949b9e864fad4d6580230b85550b9d49cbd7d50ddce`,
`4eb33d3dbc9ea5ab90004e59529e7e5a46153929ebff4cea39c41ca7b97c836c`,
and `2fba3f8274f7241941c087fccc5c17252df5c713e3aebdd368e620b7062c5c27`.

### Notebook02 cumulative mixed-T horizon estimator — 2026-09-15

The existing mixed-T R2 horizon curves used a noisy single-frame endpoint
ratio even though the final table used the frames30..199 linear strain fit.
Post-fit job4687484 completed the same cumulative normalized boundary-strain
affine fit over frames30..horizon for every horizon37..199, separately for AE
and autonomous rollout. Frame199 reproduces the existing final-fit artifact to
<1.2e-16, and before/after checkpoint hashes are identical. No fitting,
selection, or checkpoint mutation occurred. Job used mendels_q,8CPU,16GB,
placefree:shared.

Per user clarification, autonomous rollout from the one-trajectory model is
not part of the scientific comparison and is omitted from notebook02 plots and
tables; that model is used only for latent-organization interpretation. The
position-prediction diagnostic reports only the 20-per-source model. Its mixed
node AE/rollout R2 at horizons100/150/199 is .526/.591, .883/.900, and
.938/.958. Mixed stiffness AE/rollout is .613/.407, .581/.488, and .640/.527,
all80/80. Early prefix fits remain unstable because the initial fit windows are
short: displayed values are clipped at0 while CSVs preserve negatives. Low-T
curves retain the physical endpoint estimator; mixed-T curves use cumulative
fits. Notebook02 executes14cells,4inline PNGs,0errors. Exporter/builder/notebook
SHA256 are `23356303123777c30550765a06375cf27b39d5a9fc3a0c380572433d22f1e62f`,
`62251217f093d07685d5d697b04cfb66d8024db1ecb9b96d972c25d0201bfa10`,
and `29199ece806bf4303a791b0fc20014b9cc4f739c471c73626b2bb914b8ea2730`.

### All-four AE with low-T-only propagator — 2026-09-15

User identified the missing hybrid behind the historical06b interpretation:
expose the AE to both low/mixed trajectories but prevent stochastic mixed-T
increments from supervising a deterministic propagator. Freeze the completed
all4-twenty AE seed123 (SHA256
`3662a61b533761a252cec48847b476ca05bb65c79bd77489c12da89068e520fb`),
then fit the otherwise identical deltaMLP64/reference-mean16 propagator on
node_low+stiffness_low IDs0..19 and select only by low-T latent-delta validation
loss on IDs20..39. Temperature/sourceID/p-ratio/strain remain absent. Evaluate
all four heldout-development sources IDs40..119 with the endpoint metric and
the normalized boundary-strain OLS over frames30..horizon37..199.

Low/mixed same-ID trajectories share topology, but no train/validation identity
overlaps heldout IDs40..119. The AE has mixed-T exposure; the propagator does
not. In addition to mean-response p-ratio R2, predeclared post-fit diagnostics
measure true-versus-predicted transverse residual SD, its temperature
association, and framewise residual waveform correlation. These distinguish
predicting through thermal noise from predicting its stochastic realization.

Frozen snapshot `all4_ae_low_prop_1000/code_v1`, code-manifest SHA256
`44273f41d3ec5ca9666cbae12ff330b744fd4618f0ce454ce16344920326ed9a`,
protocol SHA256
`3147e788b34b6e8f5a579aff21f8368268f5d46de506119818c4509d6f17c6ea`,
and unchanged data-manifest SHA256
`c92162d0c2fd475e41f13f41596685274b346e7d81d23e9115aba317948c824a`.
Initial sandboxed qsub failed before job creation because PBS could not write
its `/var/tmp` script; frozen files were retained. Direct retry submitted fit
4687497 and afterok collector4687510, each mendels_q,8CPU,16GB,
placefree:shared,8h, without host pins. Fit4687497 completed its one-epoch smoke
propagator but failed while exporting the smoke exposure receipt because direct
MetaForge graphs lack optional `source_sim_id`; production never started and
collector4687510 cannot run. This is an identity-schema failure, not a model
result. Code_v2 derives receipt IDs from the exact frozen split, changes no
scientific setting, and uses a distinct retry output so the failed smoke is
preserved. Retry code-manifest SHA256
`9a150377314efb46ff6f201ddfddf8af22c0173f78f7be7cd35d67a676237bae`;
fit4687511 and afterok collector4687512 submitted with the same resources.
Results pending.

### Two-frame mixed-temperature extrapolation pilot — 2026-09-15

At user request, this focused pilot observes only frame0 (reference) and frame1
(first displacement), then stochastically forecasts frames2..199. Both the 2D
MP2/reference-context16 AE and Gaussian latent propagator train only on the
mixed-T node/stiffness sources at0.1K and1K. The model receives no temperature,
source label, p-ratio, or strain. The heldout-development evaluation comprises
the remaining81 IDs per source at5/10/20/30K; IDs120..199 remain untouched
final test. Because network identity changes with temperature, the declared
claim is combined temperature-and-unseen-structure extrapolation, not isolated
temperature causality.

The primary metric is decoded-position normalized-boundary-strain p-ratio R2
from frames30..199, reported separately for AE reconstruction and the mean of
8 autonomous stochastic rollouts, source-wise and temperature-wise with
valid/total counts. Predeclared fluctuation checks are transverse affine-fit
residual SD, latent-increment jitter, temperature trend, and stochastic interval
coverage. Training has29 IDs/source (17 at0.1K,12 at1K), validation has10
(4/6), and high-T evaluation counts are25/17/19/20 at5/10/20/30K.

Frozen snapshot `mixed_temperature_extrapolation_2frame/code_v1`, code-manifest
SHA256 `b71dae11f0c3a7475e78118fcde1e31bd9be0e457f34f8a3de4521d62f980809`,
protocol SHA256
`fe5ab1f20e7f0f61c998b09fe75d8b21a70395671f8ab58e5cd861436d3a208a`,
data-manifest SHA256
`c92162d0c2fd475e41f13f41596685274b346e7d81d23e9115aba317948c824a`,
and split-audit SHA256
`04ad927a797d87b731d5b86dcea15bf22d48672b6c9817a62a215b9a3326f3a7`.
Smoke4687537 gates AE4687538 and dynamics/evaluation4687539. Jobs use
mendels_q,8CPU,16GB,placefree:shared without host pins. Results pending.

Smoke4687537 completed its AE smoke fit, then exited1 before dynamics fitting:
the generic capacity loader expected a `spec` field that the standard
`ae_only` checkpoint does not contain. This is a loader-schema failure, not a
model result; the AE smoke artifact and failed log are preserved. Code_v2 uses
the matching standard AE-cache loader and distinct retry smoke paths, with no
scientific change. Its code-manifest SHA256 is
`470a6c3e0e370917c20be81c2d7f6f941f5563df7710d859eac1f2d6ef92e872`.
Retry smoke4687543 gates AE4687544 and dynamics/evaluation4687545 with the
same resources. All three completed successfully. On high-T development IDs,
AE reconstruction/8-sample autonomous rollout fitted-p-ratio R2 is
.7993/.6325 for node_mixed and .6251/-.0043 for stiffness_mixed, all81/81
valid. Source-temperature rollout R2 for node is .6804/.6086/.5693/.6315 at
5/10/20/30K; stiffness is -.0911/.2396/-.3952/.1733 with25/17/19/20 valid.
The stochastic interval contains true p-ratio for .765 node and .691 stiffness.
However, autonomous transverse-residual SD is only .481 and .393 of truth,
and latent-increment jitter only .289 and .324 of truth. Thus the two-frame
model retains useful unseen-high-T mean-response signal for node networks but
does not establish stiffness mean-response extrapolation or correct thermal
fluctuation scaling. AE high-T values are representation tests because the AE
encodes observed high-T frames; they are not autonomous forecasts. Final IDs
120..199 remain untouched. AE/dynamics SHA256 are
`f00ffb454aaa50366c1ab871633d787d1913ea7e4568f3c45968fb7b57ed93aa`
and `53625042c65cd21a8b6cd24360b6673644bb88d1d23e039385ee916d4c83dce4`.

### Executed 200-MD-step dynamics notebooks — 2026-09-16

The completed dense-cadence study is presented separately under
`notebooks/dynamics/200_md_step/`; the original notebook01-03 files remain the
1,000-MD-step study. The dense data use frames0..499 at200MDsteps/frame and end
at99,800MDsteps, whereas the originals use frames0..199 at1,000MDsteps/frame
and end at199,000MDsteps. The new notebooks are
`01_one_trajectory_per_source.ipynb`,
`02_twenty_trajectories_and_rollout.ipynb`, and
`03_latent_morph_and_attribution.ipynb`.

For the all-four 20-per-source model, frame499 low-T endpoint p-ratio R2 is
.827602/.782599 AE/rollout for node and .389430/.317415 for stiffness, all
80/80 valid. Mixed-T frames150..499 strain-OLS p-ratio R2 is
.409609/.324559 for node and .291957/.273409 for stiffness, all80/80. The
joint-low model reaches low-T .954697/.957321 node and .935150/.801898
stiffness, while its mixed-T transfer is .188018/.612441 node and
.145180/.373889 stiffness. One-per-source is approximately zero, and the
joint-mixed-only arm does not recover useful post-fit R2 on this shortened
physical horizon. The true mixed-T trajectory-fit median R2 is only .255970
node and .195556 stiffness; predicted paths are considerably smoother. This
supports the interpretation that denser sampling does not replace sufficient
total deformation.

Notebook01 has11cells/3inline plots, notebook02 13cells/4plots, and notebook03
11cells/4plots; all code cells executed with zero errors. Notebook01 executed
locally after loopback permission was granted. Shared render job4687720 exited0
for notebooks02/03 using mendels_q,2CPU,8GB,placefree:shared. Builder SHA256 is
`c215a034fe6345c8f33656f2ba8b225d209b758a8f2a90bfaa8047048f205e1f`.
Notebook SHA256 values in order are
`8615c6041abf1339fdd19433208832187678d1bd11976c39a95ff413c776df1e`,
`d72c84243660384a2c4e1e4c4bd252ac250520153c1f052646a73dc617022e4e`,
and `32e481a5258890bb08cda46521f72d72a0e500951079d78f636436485a60e653`.

### Declared first-50-frame ablation on 200-MD-step data — 2026-09-16

At user request, this matched ablation trains only on frames0..49 (MD steps
0..9,800) of the audited 200-MD-step caches, while evaluating heldout IDs40..119
through frame499 (MD step99,800). It includes the all-four one-per-source and
20-per-source arms at seeds123/456/786. Architecture, train/validation IDs,
coordinate-only objectives, uniform frame weights, optimization budgets and
checkpoint rules remain unchanged. AE limits are50frames; propagator limits
are49 one-step transitions. p-ratio and strain remain absent from inputs,
losses and selection.

Predeclared reporting separates endpoint p-ratio at frames49 and499, AE
reconstruction from autonomous rollout, and every source with valid/total
counts. Mixed-T additionally retains the original frames150..499 normalized
boundary-strain OLS as a strict extrapolation diagnostic and records a
frames0..49 in-window fit. Full latent paths for both low- and mixed-T sources
are exported from the frozen checkpoints. Seed123 gives the direct matched
comparison to the completed 500-frame training; three first-50 seeds quantify
its stability. No final-test IDs120..199 are used.

The frozen protocol is
`hpc/experiments/mini_temperature200_train50/protocol.json`. A single smoke
fit/collection gates all six production fits; each job requests mendels_q,
8CPU,16GB and placefree:shared without host pins. Jobs and frozen hashes are
recorded under `notebooks/results/mini_temperature200_train50/job_ledger.json`.

### Mixed-T 1,000-MD-step rollout metric audit — 2026-09-16

At user request, the frozen all-four 20-per-source seed123 model was audited
without changing any notebook or checkpoint. Existing cumulative prefix
artifacts show that autonomous predicted-prefix versus final true frames30..199
p-ratio R2 is already .960 at horizon37 and .958 at199 for node mixed-T. For
stiffness mixed-T it is .595 at37 and .527 at199. In contrast, same-horizon
predicted-prefix versus noisy true-prefix R2 rises .114 to .958 node and .083
to .527 stiffness. True-prefix versus final-truth R2 itself rises from strongly
negative to1.0. All comparisons use the same80/80 heldout IDs40..119.

Post-fit job4687761 then evaluated a fixed30-frame rolling strain window for
horizons59..199. Same-horizon R2 does not improve: node is .221 at59, .020 at
100 and .223 at199; stiffness is .059, .102 and -.037, all80/80. Meanwhile
rolling predicted p-ratio versus final true p-ratio remains high and nearly
constant: node .960/.959/.952 and stiffness .586/.537/.473 at those horizons.
The true rolling-window fits are intrinsically weak (median physical fit R2 at
horizon199 .123 node and .072 stiffness), whereas predicted paths are nearly
perfectly linear. Therefore the increasing cumulative-horizon R2 is dominated
by maturation/averaging of the mixed-T truth estimator, not improvement of the
autonomous rollout later in time. The rollout has largely selected its smooth
network-specific response much earlier.

Job4687761 exited0 in00:03:50 using mendels_q,8CPU,16GB,
placefree:shared; peak memory3404724KB. Checkpoint SHA256 values were unchanged
before/after: AE
`3662a61b533761a252cec48847b476ca05bb65c79bd77489c12da89068e520fb`
and propagator
`9ab4b2c4c7e630f368fc24b3a105a751bba2e97b153b212e6229cd30133dcb47`.
Audit script SHA256 is
`74e0671fd9a0fbc6b988cc203c21a867d23ad07be163d80916c8cde432d07798`;
artifacts are under `notebooks/results/all4_coverage_1000/rollout_metric_audit`.

### Mixed-T 200-MD-step rollout metric audit and notebook update — 2026-09-16

The same post-fit decomposition was applied to the frozen all-four
20-per-source 200-MD-step model, all80/80 heldout IDs40..119. Cumulative
same-horizon rollout R2 rises from -.015/.010 at horizon157 to .325/.273 at
499 for node/stiffness mixed-T, but the true prefix versus final truth estimate
also matures from extremely unstable values to1.0. Rollout-prefix versus final
truth is already .466 node and .101 stiffness at157, becoming .325 and .273 at
499. Thus node does not genuinely improve with horizon; stiffness gains some
final-response information, but far less than the raw cumulative curve implies.

A fixed150-frame window spans29,800MDsteps, matching the physical duration of
the 30-frame window in the 1,000-MD-step audit. Same-horizon R2 at horizons299/
400/499 is .068/-.164/-.085 node and -.001/.121/.109 stiffness. Fixed-window
rollout versus final truth is .404/.295/.202 node and .179/.286/.312 stiffness.
The cumulative rise is therefore again dominated by the improving truth
estimate and growing averaging window. The physical rolling fits remain weak
at frame499: median truth fit R2 .089 node and .184 stiffness, versus essentially
1.0 for the smooth predicted paths. Checkpoints were hash-identical before and
after. Audit script SHA256 is
`2e1230df3ccd146dba180b0b2c1dec8d488e0e2b90a37768c23ff273aedc255a`;
artifacts are under
`notebooks/results/mini_temperature200_frames500/rollout_metric_audit`.

Frozen-checkpoint jobs4687754/4687755 exported complete mixed-T latent paths
for the one-/20-per-source models: two sources x80 heldout trajectories x500
frames each, with output SHA256 values
`6f4d001535c538d50c2ab2ad3c2ef25594b81bd657923b5b24a06010411fc170`
and `3e33f99a6279fc7a283fb469ced762483707d4b0c9b9f285c9d56e3476718e02`.
The 200-MD notebook01/02 latent displays now use these mixed-T paths. After the
audit, the user requested a simpler presentation: notebook02 for both 1,000-
and200-MD studies now shows endpoint p-ratio R2 against MD step with prediction
and truth evaluated at the same frame, followed by the final network-wise
scatter. The estimator-maturity/fixed-window panels were removed from the
notebooks but their frozen audit artifacts above remain preserved. Curves clip
negative R2 visually at zero while the endpoint tables retain exact values.
Both revised notebook02 files executed with every code cell complete and zero
errors. Their SHA256 values are
`2d136ae37d38e838c842fb1a2ca9f8431a755a8c8dad941f1d2d729c0a78bcac`
(1,000-MD) and
`c2a375e7c6b7f7c65c4fae189fcfd21192c8ae3ac9464315f6dd00f8a950fdad`
(200-MD). The 1,000-MD latent-path display uses a centered three-frame vector
average, keeps frames0/199 exact and shows the raw path faintly; this supersedes
the global quadratic z0/z1 display without changing any metric.

The mixed-T network-wise scatter plots in the 1,000-MD notebook now use a
shared logarithmic temperature colour scale (0.1--30K), with source/model
identity retained by subplot or marker. Temperature comes from the hashed
MetaForge manifests and remains a display-only diagnostic; it is not supplied
to either model. The revised notebook executed all8 code cells with zero errors.

The corresponding final frame-499 true-versus-predicted p-ratio scatter in the
200-MD notebook now uses the same shared logarithmic 0.1--30K temperature
colour scale. Sources remain separated by subplot; the endpoint estimator and
frozen predictions are unchanged. Temperature is joined
from `mini_temperature200_frames500/data_manifest.json` by source and trajectory
ID and is display-only. The revised notebook executed all6 code cells with zero
errors; SHA256
`3bd036a26c61c9631f224bf89f93ea5cae049d0fd726c25dd07546471611e792`.

### Mixed-T initial-latent anchor pilot -- 2026-09-16

At user request, one seed tests whether explicitly retaining frame-zero latent
identity improves the underperforming stiffness mixed-T propagator. Baseline is
the completed mixed-only seed123 model with frozen AE SHA256
`358a57bf22e5fb7548ebc5375ae54efdb2a037d8f76ceb4cf69be5e127d1f8a6`.
The intervention appends the normalized frame-zero2D latent after the unchanged
mean16 static-context projection at every delta step. Current-z input, static
context, splits,20 trajectories/source, frames0--199, one-step delta loss,
optimizer and validation latent-loss selection remain fixed. The first MLP
layer gains exactly128 parameters; local configuration, forward-shape and
parameter-difference checks passed. No p-ratio, strain, temperature or source
label enters fitting or selection.

The single predeclared anchored run uses seed123 and a smoke gate. Heldout
development IDs40--69 (30/source) are evaluated regardless of outcome with
post-fit normalized boundary-strain OLS frames30--199 as the primary p-ratio
R2 and physical frame199 endpoint R2 as secondary. PBS fit job
`4688055.zeus-master` is running; dependent collector `4688056.zeus-master` is
held afterok. Both request mendels_q,8CPU,16GB,placefree:shared without host
pins. Frozen snapshot and ledger:
`notebooks/results/mini_temperature1000_initial_anchor_pilot/`; code manifest
SHA256 `62f6643363f34c33662ddc4193f94475486bbe5e48913059651ef89be812c87f`.

### Dynamics code and notebook simplification -- 2026-09-16

No model, checkpoint, estimator, split, or result artifact changed. The active
API now exposes one recommended pair through `lss.dynamics.canonical`: the
message-passing reference16 AE (2D by default) and the one-step delta-MLP with
mean16 static context. The historical configurable implementations retain their
old `lss.dynamics.training` and `lss.dynamics.experiment` import paths through
thin compatibility modules and are isolated under `dynamics/_compat/`.
Completed direct/factorized/path/static, broad-trajectory, local-dynamics,
observed-history/acceleration and radical propagator studies moved under
`lss.past_experiments.dynamics`; their old module paths remain compatibility
imports for frozen runners. After the moves, focused compatibility coverage
passed24 tests and22 subtests, and the final full suite passed92 tests and28
subtests.

Paper-facing notebooks01--03 for both1,000- and200-MD cadence now use shorter
section and plot titles. Both notebook02 variants display autonomous rollout
only: AE reconstruction curves, scatter panels and tables were removed from the
presentation while their frozen result rows remain untouched. All six notebooks
executed with every code cell complete and zero errors. SHA256 values are
`2e862b23e19d575385533a2089c8938a337ac565383d6e077ab93582be744639`,
`cf86a0ca765ed06d388f4dfc95f8fa7b319c56fd3482c992af5a87bfff3bc255`,
`a717c5e393bb2290a4fb7e5659fdb8aa647dc5d172d7e4352686872214eb6c1e`
for the1,000-MD notebooks01--03 and
`7438858d16b1b3fdc27201dd6b7d57a95b6a40f563594ced858544601e75e51a`,
`89fd6d006de25acf18fcc4c8bbc59783911d0ece1fadf407a66440a82ffde255`,
`ca04beb809ae0247920e1b2ad2076ba849f36fc5b9578b204a923f34fcf9ec77`
for the200-MD notebooks01--03.

Notebook03's paper-facing mechanism analysis now uses only the held-out
stiffness-optimized low-T source (IDs40--119;80/80 for each frozen one- and
twenty-per-source model). Final response is the physical frame199 endpoint
p-ratio, not the mixed-T fitted estimator. Gyration radius and position
anisotropy were removed from the static-descriptor panel; all displayed
correlations are absolute. The mixed-T feature-attribution panel was also
removed, and the edge rendering now uses the frozen low-T attribution artifact.
The rebuilt notebook completed all7 code cells with zero errors.
Its initial-latent panel now overlays node- and stiffness-optimized low-T
networks on the same z0/z1 axes, with marker denoting family and colour denoting
physical frame199 p-ratio. The later mechanism correlations remain focused on
stiffness-optimized low-T trajectories.

### Raw two-frame temperature-rollout p-ratio export -- 2026-09-16

Notebook08 is being replaced with a raw physical frame-wise p-ratio analysis:
frame199 true-versus-autonomous scatter, p-ratio R2 versus frame, per-temperature
endpoint statistics, and representative true/predicted trajectories, with
temperature encoded consistently in every panel. Prediction is computed from
the mean decoded physical positions across the same eight frozen stochastic
rollouts; no trajectory line fit enters these panels. This is post-fit
evaluation of the completed seed123 AE/dynamics checkpoints, not retraining or
selection, and final IDs120--199 remain untouched. Initial job4688130 failed
before inference because it used an environment without graph_utils; its log is
preserved. Corrected job4688131 used mendels_q,8CPU,16GB,placefree:shared and
completed exit0 in00:15:06. Raw frame199 R2 is.612353 for node_mixed and
.089240 for stiffness_mixed,81/81 valid each. Node temperature-specific R2 is
.637331/.628417/.670464/.511393 at5/10/20/30K; stiffness is
-.008409/.297348/-.288063/.145136. The executed notebook completed5/5 code
cells with zero errors; SHA256 is
`8b48c0548d993f5754cd5a761c468ab76c9642b472fe0415971feab6ab20f86e`.
Export-script SHA256 is
`12382cb4c630c222ab054d974a5030a2edf5cabfef85f88143608371dc60d5e7`.

### High-temperature AE to low-temperature transfer -- 2026-09-16

This new smoke-gated three-seed experiment tests whether a2D reference-MP2 AE
trained only on noisy20K/30K node- and stiffness-optimized trajectories retains
the clean low-T response. Training uses20 IDs/source (10 per temperature),
validation uses10/source (5 per temperature), and checkpoint selection remains
coordinate reconstruction MSE only. Frozen evaluation covers the20 matched
low-T structures and9 wholly unseen low-T structures, reporting primary
post-fit p-ratio R2 over frames30--199 and secondary raw frame199 endpoint R2,
source-wise with valid/total counts. Temperature, source label, p-ratio and
strain remain absent from inputs/losses/selection; final IDs120--199 are
reserved. Smoke job4688140 gated seed123/456/786 jobs4688141--4688143.
All use mendels_q,8CPU,16GB,placefree:shared. Protocol/script
SHA256 values are
`29a348372ab66f2ce4770378505cf2a258d709789cf98818e1fa0da85dc7b9de`
and `744e9534bac3a7a5103b28eaaae40944bc053265a5fadcc8fffa8976fc6ec788`.

Smoke4688140 subsequently passed the one-epoch training and both matched/unseen
low-T evaluation paths with exit0 in00:02:47; the three production seeds were
released and are running. Notebook01's initial-latent/p-ratio diagnostic now
uses one panel per latent coordinate and overlays node/stiffness mixed-T
heldout points, source-specific trend lines, and both actual training markers.
Its displayed Pearson and Spearman correlations are absolute.
# Five-seed p-ratio estimation at 200/1000-MD cadence — 2026-09-16

User generated five matched thermal-velocity replicas for every low/mixed,
node/stiffness dataset at both 1,000-MD-step spacing (200 frames) and
200-MD-step spacing (1,500 frames). MetaForge arrays 4688160[]/4688163[] and
their validation collectors 4688161/4688164 were still running/held at
submission. A post-fit-only analysis array 4688169[] is held after both
collectors; collector 4688170 and notebook-09 finalizer 4688171 are chained
after it. All use mendels_q, place=free:shared, no host pins. Exact resources,
dependencies, code hashes and job identities are in
`notebooks/results/five_seed_pratio/job_ledger.json`.

The predeclared primary estimator does not average five noisy ratios. It fixes
q=.10 boundary-node groups from the same network's first low-T frame, computes
normalized x/y boundary strain relative to each replica's own frame 0, removes
one intercept per thermal seed, and pools a single OLS transverse-versus-axial
slope across all five replicas; p is minus that shared slope. The analysis also
retains five separate slopes with sample SD/SEM/95% t interval, the
ensemble-mean-trajectory slope, and raw ensemble endpoint p-ratio. This is
compared against recorded source response and source-rollout response with
source/temperature valid-total counts. The 1,000-MD window is frames30..199.
The 200-MD analysis reports frames150..499, a cadence-matched physical horizon
frames150..995 (30,000..199,000 MD steps), and frames150..1499. Prefix summaries
every10,000MDsteps will test whether apparent late R2 improvement is signal or
endpoint noise. No p-ratio/temperature/source label enters training or model
selection; no model is retrained, and no final-test split is used. Shared-slope
unit tests passed3/3. Results are pending.

### 200-MD stride-five propagator pilot — 2026-09-16

User requested trying a larger dynamics step after the first-50-frame dense
training failed even near its training boundary. This pilot isolates transition
signal-to-noise without retraining the representation: the completed all-four,
20-per-source 500-frame AE is frozen byte-identically, and only propagator
`frame_skip` changes from1 (200MD) to5 (1,000MD). Training/validation sources,
IDs, architecture, static context, delta loss, optimizer,60epoch/patience12
budget and validation latent-loss selection are unchanged. Each trajectory now
supplies99 transitions 0→5 through490→495. Temperature, source label, strain
and p-ratio remain absent from inputs/losses/selection.

Three seeds123/456/786 are gated by smoke4688462; production jobs4688463--
4688465 are held afterok, and collector4688466 is held after all three. All
fits/evaluations use mendels_q,8CPU,16GB,place=free:shared; collector uses
1CPU/4GB shared. Heldout-development IDs40..119/source are always evaluated.
Matched reporting uses low-T physical endpoint p-ratio R2 at common stored
frame495 and mixed-T normalized-strain OLS on common stored frames150,155,...,
495, compared with the existing stride-one checkpoint on exactly those frames.
Exact hashes, dependencies and resources are in
`notebooks/results/mini_temperature200_stride5_pilot/job_ledger.json`.
Results pending; no final-test IDs are used.

Smoke4688462 subsequently completed all four training/evaluation source paths,
kept the frozen AE hash unchanged, and exited0 after about2.5minutes. Production
seeds4688463--4688465 were released and verified running; collector4688466
remains held afterok. Smoke propagator SHA256
`d572b64dcd797c5f6f3d050af1cf97a06237ea4ea59a8ba8a91370aa87ec3982`.

Notebook02 now adds one response-blind matched low-T latent-path comparison for
10 evenly spaced heldout IDs40..119 in each node/stiffness family. Each identical physical
trajectory is encoded by the frozen one- and20-per-source models. At user
request, the final display overlays each encoder's raw z0/z1 outputs without
centering, scaling, rotation or alignment. Their independently learned absolute
offsets and orientations therefore are not physically comparable. All four
groups now share one axis: the1-per-source paths remain orange, while only the
20-per-source paths use the common frame199-p-ratio heatmap. Solid/dashed lines
denote node/stiffness optimization, and frame0/199 are filled/open.

Notebook02 now also contains a compact endpoint-movement comparison using the
same frozen seed123 models and all80/80 heldout low-T IDs40--119 per family.
Endpoint rays are centered at zero, rotated so each model/source mean direction
is horizontal, and normalized by that cohort's median displacement magnitude;
this response-blind display exposes direction spread without comparing arbitrary
raw latent axes or scales. The final presentation uses a shared equal axis scale
across the four training/source ray panels, so the near-zero transverse spread
of one-source training is not magnified; endpoint colour denotes frame199
p-ratio. Paired bar panels show initial-coordinate, movement-distance and
movement-coordinate absolute Pearson correlations. In the one-per-source model, absolute Pearson
correlation of movement distance with frame199 p-ratio is.845 node and.115
stiffness, while absolute correlations for delta-z0/delta-z1 are.848/.844 and
.019/.140. In the20-per-source model, distance correlation is.587 node and.116
stiffness, while delta-z0/delta-z1 correlations are.977/.982 and.886/.978.
Thus one-source paths are essentially a common direction with varying distance;
twenty-source paths spread in direction, and stiffness response is represented
by the two-dimensional movement components rather than total distance. This is
descriptive frozen-model analysis, not response-supervised fitting or selection.
The executed notebook has20cells,10code cells and zero errors. Updated
builder/notebook SHA256 are
superseded by the final hashes below.

Notebook02 now tests whether the frozen20-per-source encoder uses a shared
low-T response representation across node- and stiffness-optimized families.
For all heldout IDs40--119, the left panel shows raw endpoint delta-z0/delta-z1
with source encoded by marker and physical frame199 p-ratio by colour. A single
post-hoc affine p-ratio readout fitted on both families gives R2=.971 over
160/160 networks. The stricter cross-family test fits this two-coordinate
readout on one family only and applies it unchanged to the other: stiffness to
node gives R2=.942 and node to stiffness R2=.917, each80/80. The transfer
scatter reveals a modest family-dependent calibration offset, so the supported
claim is a shared latent response coordinate, not an exactly identical mapping.
This response-calibrated linear readout is diagnostic only; it did not enter AE
training, loss, checkpoint selection or autonomous position rollout. The
corresponding builder/notebook hashes are superseded below.

Notebook02 now adds a negative-response early-turn zoom for the frozen
20-per-source encoder. It preselects the six most negative frame199-p heldout
networks in each low-T family and shows each complete raw-z path beside frames
0--20 translated by its own frame-zero z only; there is no rotation or scaling.
The same figure now adds a source-wise frames1--20 overlay of the six-network
median latent turn angle relative to the eventual frame199 direction and the
actual cumulative framewise physical p(t)=-deltaLy(t)/deltaLx(t), with separate
degree and p-ratio axes and10--90% bands. This compares temporal shape without
claiming that either latent coordinate equals the physical ratio.
Across all80/80 heldout networks/source, the frame1 angle relative to each
trajectory's eventual frame199 displacement has median46.0degrees for negative
versus3.1degrees for nonnegative node networks, and34.1 versus5.3degrees for
stiffness networks. Thus the strong initial bend is specifically associated
with negative response in this frozen seed, while the endpoint delta-z remains
the separate final-response summary. This is descriptive heldout analysis, not
checkpoint selection. The executed notebook has24cells,12code cells and zero
errors. Final builder/notebook SHA256 are
`46b1668eb780b82b8e57e77f22e92f810a8ba342736b72778fdfabbbb5dea2e4`
and `227da9d2409d9e433dea471312e19d7fdf8efbd95d33fe84d90fc72a9d640452`.

At user correction, notebook02 removed the entire normalized endpoint-ray and
bar section `Movement: one versus twenty`; its results remain historical above
but are no longer presented. The negative-p early analysis also removed the
abstract turn-angle proxy. For the same six preselected most-negative heldout
networks/source, frames0--20 now show (1) delta-z1 versus delta-z0 from each
network's own frame0, (2) the measured delta-Ly versus delta-Lx physical path,
and (3) the direct cumulative latent slope delta-z1/delta-z0 against physical
p(t)=-delta-Ly/delta-Lx. Both paths share the framewise-p colour scale. Median
within-network Pearson r between these two slopes is.950 node and.990
stiffness,6/6 each. An independent all-heldout check gives median r=.974/.991
over80/80 node/stiffness trajectories across frames1--20; fixed-frame pooled
correlation across both families is.948,.977,.930 at frames1,5,20. Thus the
early latent curve is most directly described as the evolving physical
deformation ratio, while neither raw latent coordinate individually equals p.
Notebook02 executes22cells,11code cells,zero errors. Final builder/notebook
SHA256 are
`d4fad92790ecbefd1a531a0b4d7a6f8bc81355f82ee5454b9c73487d3d5422fb`
and `c0d72f8d2378a731d3eab3bcd80e1fc81424a44a077f4fa71dac786a5d518488`.

Notebook02 now replaces the weaker component overlay with the closest simple
physical matches found by a systematic scan of the frozen seed123,
20-networks-per-source low-T heldout trajectories (IDs40--119, frames1--20,
1000 MD steps/frame). Within each network, delta-z0 follows the deformation
difference delta-Ly-delta-Lx with median absolute Pearson r=.9999 for both
node and stiffness sources; delta-z1 follows the deformation sum
delta-Lx+delta-Ly with median absolute r rounding to1.0000 for both sources
(80/80 each). The ratio relation remains strong but visibly nonlinear:
median r(delta-z1/delta-z0, -delta-Ly/delta-Lx)=.9739 node and.9910
stiffness (80/80). The inline plot normalizes each displayed curve by its own
frame20 value, so it establishes near-identical temporal shape rather than
equality of units or a universal cross-network calibration. Notebook execution
completed with22cells,11code cells, zero errors and zero warnings. Final
builder/notebook SHA256 are
`a36162164a4df10a961f6623047b8fbd850d8ad912e7a595555bcbeb2176a492`
and `41897017295f098d0fb5f1953b697a1d79dc12d36e8a0fb6eb54cd60abcd631d`.

Notebook02 now exposes all six selected negative-p trajectories individually in
the early physical/latent shape comparison, using one colour per network and
solid/dashed styles for latent/physical quantities. A third column shows raw,
unscaled delta-Lx, demonstrating that it is not identical across networks. Over
all80 heldout networks/source and frames0--20, every source has80 distinct
delta-Lx curves; the maximum same-frame across-network range is.00952 for node
and.02047 for stiffness. Among the six displayed networks the corresponding
maxima are.00771 and.00824. What is nearly shared is the normalized temporal
shape, not the delta-Lx magnitude. Notebook execution remains22cells,
11code cells, zero errors and zero warnings. Final builder/notebook SHA256 are
`36ade3ac6c41f9fe9014d9b78f5009cebdc9b9c5d9410faa06bf0a17da47440f`
and `ec3bc182838e3b5b2cdb604f462732c129f8f70850c90d575ef631cf386cf00b`.

Notebook02 now distinguishes subtracting versus normalizing the initial box.
The cached x_deformation already equals Lx(t)-Lx(0) using fixed frame-zero
boundary nodes. Initial Lx nevertheless varies from9.73--13.86 for node and
9.85--19.64 for stiffness. Dividing by each network's own Lx(0) collapses the
six displayed compression curves strongly. Across all80/source over
frames0--20, the maximum same-frame raw delta-Lx range is.00952 node/.02047
stiffness, whereas the delta-Lx/Lx(0) range is.000316/.000243. The notebook
shows raw and normalized curves side by side. Widths were recovered from the
exact hash-verified source datasets with fixed quantile-.10 boundary nodes and
cached as160 unique rows in `test/initial_box_widths.csv`.

Notebook02 also adds the requested centered-change diagnostic, defining
delta-z(t)=z(t+1)-z(t-1) and applying the same centered operation to physical
deformations at frames1--19. Median within-network absolute correlations over
80/80 trajectories are .992/.935 for delta-z0 versus centered
(delta-Ly-delta-Lx), .987/.976 for delta-z1 versus centered
(delta-Lx+delta-Ly), and .963/.972 for their latent/physical ratios
(node/stiffness). Pooled across networks, the component correlations fall to
.860/.016 and.862/.372, showing network-specific component calibration; the
centered ratio is more consistent across networks within each family at
.882/.976. Thus the derivative test supports local physical tracking within
each trajectory but not a universal raw-coordinate identity, especially for
stiffness. Notebook execution has
22cells,11code cells, zero errors and zero warnings. SHA256 for the width
collector, width cache, builder and notebook are respectively
`0b709ec05b153b7de997f2217353ded804b8ad244ee21f859f6e99a0c8e99391`,
`c776272d2c35ba94caee278d8616c9c70532f00f85e9bc91488af9d2d17cb8bd`,
`04111f87835fa8c123b5f998cc3987373a0ca1b1b0cde676fb23651f271fbeb4`
and `40f6117f192b81a4bd320b68716f59bf4a3b2cc1ed3543a6376bb423c46072ef`.

Follow-up frozen-data diagnostic: OLS slope of x deformation versus frame does
not correlate with frame199 low-T p-ratio once each network's initial box width
is accounted for. Over frames0--20, Pearson r for the strain slope
(delta-Lx/Lx0)/frame is.016 node and.026 stiffness (80/80); over frames0--199
it is-.032 and-.011. The raw stiffness delta-Lx slope has r=-.266, but this
disappears after Lx0 normalization and is therefore an initial-size artifact.
In contrast, the frames0--20 transverse strain slope
(delta-Ly/Ly0)/frame correlates strongly with final p-ratio: r=.997 node and
.993 stiffness (80/80). This is consistent with x compression being imposed
while the y response carries the mechanical variation.

A stricter cross-family calibration test shows that the centered latent ratio
must not yet be called fully transferable. Fitting one affine map from
delta-z1/delta-z0 to -delta-Ly/delta-Lx on all node low-T centered samples and
applying it unchanged to stiffness gives R2=.698; fitting on stiffness and
applying to node gives R2=-.141. The asymmetry reflects the narrower stiffness
latent-ratio range (SD.360 versus1.044 for node), so the reverse direction is
an extrapolation. A pooled two-family affine calibration gives source-wise
R2=.768 node/.748 stiffness. The supported claim is a shared, strongly ordered
instantaneous-response signal with family/range-dependent calibration, not a
universal transferable equality.

At user request, notebook02 removes the deformation-sum/difference overlays and
compares both latent coordinates directly with delta-Ly. For frames1--20,
median within-network absolute correlation is .9939/.9965 for delta-z0 versus
delta-Ly and .9992/.9986 for delta-z1 versus delta-Ly (node/stiffness,80/80).
However, the frame20-normalized curves do not coincide: high correlation here
largely reflects shared monotonic progress. The centered-change diagnostic is
therefore retained as the stricter test. For delta-z(t)=z(t+1)-z(t-1), pooled
absolute correlation with centered delta-Ly is .959/.555 for z0 and.956/.903
for z1; median within-network values are.990/.940 and.988/.981. This simpler
view shows that z1 is the more consistently shared transverse-response
coordinate, while z0 calibration is family dependent. The physical-ratio panel
is retained separately. Notebook execution has22cells,11code cells, zero
errors and zero warnings. Final builder/notebook SHA256 are
`6f56fdb4c37f008d0ff5f8415093e293fa64d009f9b45feca5d9075039069a69`
and `48a05c00f1a4623007da94690e5a9c0787714271a56bdf1e99627c1dece3358d`.

Notebook02 now presents both interpretations side by side rather than choosing
one: delta-z0/delta-z1 versus delta-Ly directly, and delta-z0 versus
delta-Ly-delta-Lx / delta-z1 versus delta-Ly+delta-Lx. All four trajectory
panels divide each curve by its own frame20 value and therefore compare only
temporal shape, not magnitude or a universal calibration. The direct delta-Ly
correlations remain .9939/.9965 for z0 and.9992/.9986 for z1, whereas the
x-adjusted combinations reach .9999/.9999 and approximately1.0000/1.0000
(node/stiffness,80/80). The centered-difference scatter remains raw and
unnormalized. Notebook execution has22cells,11code cells, zero errors and zero
warnings. Final builder/notebook SHA256 are
`40f1980a4168f55650dff92a1ea4b28697d2d3310d81a2c7fca45513c98551da`
and `bf0a541a5ed07ffd9d6b944648366ad1472682027b3e7667ab00cdfac8a3bcb9`.

## Cold-rubber global-context pilot — 2026-09-17

Completed a separate small simulator study on the original serialized files in
`/rg/mendels_prj/s.sergey/graphs_cold`. The directory has2300 `.pt` trajectories;
`clean_400.json` declares1706 complete trajectories. Each file is a list of400
PyG frames with252--588 nodes, fixed topology/IDs, dynamic3D positions, five
edge channels and a centered orthorhombic box. The first three edge channels
are minimum-image source-minus-target vectors, followed by length and stiffness.
No explicit plastic-event frame or label exists, so no event target was invented.

Frozen NumPy split seed20260917 selected12train/4validation/8final-test IDs from
`clean_400`; model seeds123/456/786. Frames1and2 are observed and frames3--399
are autonomous. Inputs are reference/current positions and minimum-image velocity
normalized only from training data, atom type, and rebuilt dynamic edge features.
Targets/checkpoint selection use positions/box dynamics only; p-ratio and strain
never enter fitting or selection. Matched processors have four64D MPNN blocks;
the global arm inserts two four-head full-node attention layers between the first
and last two blocks. Full attention is appropriate for the observed maximum588
nodes. A targeted follow-up kept the global architecture and trained through
20-step autonomous windows, selecting mean validation state MSE at horizons
20/50/100. It was the only post-pilot training change.

Smoke jobs4689383/4689384 and production jobs4689386--4689391 all exited0.
Periodic-coordinate evaluation job4689596 exited0. Multistep smoke4689599 and
production jobs4689640/4689641/4689642 exited0. Two pre-submission attempts
created no PBS job: local qsub could not write `/var/tmp`, and the login-node SSH
fallback lacked a UID mapping. Production one-step peak memory was0.96GB and
walltime5:07--15:08; multistep peak memory1.22GB and walltime3:32--4:47.

The initially predeclared raw affine diagnostic was invalid when a predicted node
crossed a periodic boundary; those unaltered results remain in `aggregate.csv`.
Frozen checkpoints were re-evaluated without retraining or reselection by mapping
each predicted and true node to its nearest reference-frame periodic image before
the same independent affine fits. This is the reported physical-coordinate
estimator. At frame399, final-test mean±SD position MSE is
`.021110±.034326` local one-step, `.000597±.000410` global one-step, and
`.000208±.000004` global multistep. Thus global context and multistep training
substantially stabilize coordinates. They do not solve response prediction:
y/x p-ratio R2 is `-1.471±1.805`, `-6.300±9.352`, and `-2.241±3.215`;
z/x is `-2218.516±3736.092`, `-2.716±1.937`, and `-.868±.425`, respectively.
Every component has24/24 valid network-seed predictions. The best isolated final
multistep y/x seed is only `.030`; no reliable p-ratio claim is supported.

Results and ledgers are in `notebooks/results/graphs_cold_global/`, including
`periodic_aggregate.csv`, `multistep_aggregate.csv`, per-network/per-frame CSVs,
checkpoints, recipes and exact trajectory hashes. Executed notebook:
`notebooks/dynamics/cold_graphs/01_global_context_rollout.ipynb`. Frozen SHA256:
loader `1648ff8d...e9894`, model `7e5fd561...ed9f`, one-step runner
`fc7aaf64...5997`, multistep runner `da1c31e2...6e0`, periodic evaluator
`109b4f1a...6ca`; full hashes are in each recipe and job ledger. Notebook has
11cells/5code cells, all executed with zero errors; notebook SHA256
`ccafcc850b7a551f88157b0190444eb1ba3f64b56e8d3d5f7b64da235789f7fd`.

## Cold-rubber fractional/contact and anchored-state validation screens — 2026-09-17

Two validation-only follow-ups were completed after the global-context pilot.
Frozen NumPy split seed `20260918` contains 48 train, 12 validation and 12
untouched final-test trajectories, excludes all 24 earlier pilot trajectories,
and has manifest SHA256
`1a0e8ffb6e8a5aef9339009acca7a394cabada3c70ee5ba0e667268e0c765541`.
Only seed 123 was used for these screens; no seed-robust or final-test claim is
made. Frames 1--2 are observed and frames 3--399 are autonomous. Neither
p-ratio, strain nor event labels enter inputs, objectives or checkpoint
selection. All reported validation p-ratio entries are 12/12 valid.

The v2 screen used fractional coordinates, prescribed constant-velocity x-box
loading, learned yz-box acceleration, and 20-step autonomous fine-tuning. It
compared permanent bonds with the same model plus dynamic minimum-image
nonbonded contacts at cutoff 1.5. Smoke jobs 4690044/4690045 and production
jobs 4690046/4690047 exited 0. Production walltime/peak memory were 14:48/1.49
GB and 35:16/3.36 GB. At frame 399, bond/contact physical minimum-image
position MSE was `.0001366/.0001214`; y/x R2 was `-.155/-.319`, and z/x R2
was `-1.641/-.687`. At frame 100 the respective y/x values were `.317/.199`
and z/x `-1.318/-.989`. Thus fractional coordinates and contacts made rollout
coordinates stable, but did not recover mechanical ranking. A frozen replay
showed high teacher-forced one-step p-ratio ceilings in the preceding models
but immediate autonomous response drift. In v2 specifically, frame-399 global
box response was already inaccurate; fractional-position drift further harmed
z/x. Results and exact checkpoint/trajectory hashes are under
`notebooks/results/graphs_cold_global_v2/`, with diagnostics in its
`diagnostics/` subdirectory. Frozen protocol/runner/model SHA256 are
`e5eedda5fb8f68f25958e7259aad2a0ae6c4d07429dbccf473e7e585a91f141a`,
`efec78ad5f69688579bf81e401d866b3a9c56a3005c7027c8925af4dedd27fce`
and `6207d9204ec7de940cc4384213ed1fbf6f47be1698b0ec6ef7dac833e16172da`.

The v3 screen replaced integrated accelerations with reference-anchored total
fractional/node and yz-box displacement. It compared contact models without
and with recurrent memory. Before results, promotion required frame-399 R2 at
least .3 on each transverse axis, mean R2 at least .5, and position MSE at most
`.001`; checkpoints were still chosen only by state MSE. Smoke jobs
4690059/4690060 and production jobs 4690061/4690062 exited 0. A submit-ledger
path error occurred only after both smoke qsub calls; exact job IDs and hashes
were recovered without duplicate submission and the path was corrected.
Production walltime/peak memory were 22:30/5.80 GB and 32:30/5.42 GB. At frame
399, nonrecurrent/recurrent MSE was `.0001353/.0001401`, y/x R2
`-5.485/-8.488`, and z/x R2 `-8.829/-12.907`; neither passed promotion.

Frozen v3 replay then separated predicted box deformation from node residual
affine modes. Removing all constant/reference-affine fractional residual modes
made position p-ratio equal box-only p-ratio but did not improve it: frame-399
box y/x and z/x R2 were `-5.576/-9.982` nonrecurrent and
`-8.171/-12.528` recurrent. The position-versus-box p gaps were only
`.00345/.00341` and `.00658/.00249`. The failure is therefore the learned
global box/affine response, not recurrent memory or late node-residual leakage.
As a physical ceiling check on the same validation data, using the true box
with purely affine reference positions gives frame-399 MSE `.0001207` and
position-derived p-ratio R2 `.902/.914`; learning global box response is a
well-defined next target. V3 artifacts and diagnostic hashes are under
`notebooks/results/graphs_cold_global_v3/`. Frozen protocol/runner SHA256 are
`fcaf37abff18af39c998431e49047319541eded1ad86ce8ca1eebab22ef973b5`
and `52d0e0b37f2dedcfb82062c3131536cad6865235fa89a61f5668ebbeaa01754a`.

## Cold-rubber direct-response iterations — 2026-09-17

All iterations below retained the same disjoint48/12/12 v2 split and seed123,
used only positions, graph structure and box/loading state for fitting, and
kept the12-network final-test split unopened. P-ratio and strain were absent
from inputs, losses and checkpoint selection. Every reported validation
p-ratio component has12/12 valid networks. These are single-seed screens, not
seed-robust final-test claims.

V4 tested static reference-graph mean/token attention encoders that directly
predicted total yz-box displacement under prescribed x loading. The initial
smokes4690067/4690068 exited1 after unconstrained one-epoch predictions made a
box non-positive. Positivity-constrained smokes4690069/4690070 exposed a
one-graph normalizer problem; corrected four-graph smokes4690073/4690074 and
production4690077/4690078 exited0. Full-horizon confirmation smokes
4690080/4690081 and productions4690083/4690084 also exited0. At frame399 the
full-horizon mean/token variants had position MSE `.00012939/.00012935` and
y/x R2 `-3.546/-3.561`, z/x `-5.469/-5.403`. Training-network replay was also
poor, so the failure was optimization/representation rather than only heldout
generalization. A coordinate-only OLS loading control reached frame399 R2
`.015/-.071` with position MSE `.00012234`, establishing a stronger neutral
baseline. Frozen final V4 protocol/runner SHA256 are
`d80c8f070dc39d7e8eba649deed79c3fce1d6b5120f0d5f06a20e9d8197541c4`
and `e6ee71d753da9b4f2e34a144d8ad6b39fc933eb0644218ae9b3e5b17617db832`.

V5 added a frozen train-fit OLS response and a zero-initialized token-GNN
residual, scaled by train residual variance. Smoke4690086 and production
4690087 exited0; production used1.13GB and1:11. The selected epoch6 checkpoint
had frame399 MSE `.00012232`, yz-box MSE `.00002355`, and p-ratio R2
`.042/-.053`. Residual training MSE stayed near one, so the GNN learned little
beyond OLS. A separate fixed-alpha structure/initial-motion ridge screen also
failed, and a temporal-slope evaluation of the frozen predictions remained
negative; estimator noise did not explain the failure. Frozen V5
protocol/runner/model SHA256 are
`86167d10c9a4ae49d2ddf556fa261bdc4430a8cb0208a17c0f7922711c45265b`,
`cbff012734e76e5c5357a4826e84c3e0bfe65921ee887325b254329e6cff8c1c`
and `6337921c191756548df86a8b826b9d61082798940298851043f175d2973676db`.

V6 changed only optimization: deterministic per-epoch graph shuffling and
balanced eight-trajectory gradient accumulation. Smoke4690091 and production
4690092 exited0; an incorrect submit-ledger hash path failed only after the
smoke qsub succeeded, so its job identity/hashes were recovered without a
duplicate submission and the path was corrected. Production used1.17GB and
2:40. The selected epoch77 checkpoint improved frame399 yz-box MSE by4.8% to
`.00002242`, with position MSE `.00012224` and p-ratio R2 `.089/~.000`.
Training residual loss fell only1.3%; the predeclared response gate still
failed. Frozen V6 protocol/runner SHA256 are
`2eb22391138c13279d2661f6fa157c73927a52e625256c679d79e84119840bfd`
and `2072e10496f9ba961bc07078ad2e72b328316fb56583dc23d6991e9425157ad2`.

At user direction, V7 made frame50/100 acceptable endpoints. It conditioned a
short-horizon token GNN on the reference graph plus the observed frame0-to-1
fractional node motion and full box velocity, then directly predicted frames
3--100. Smoke4690095 and production4690096 exited0; production used1.18GB and
2:12. The selected epoch79 checkpoint substantially improved yz-box error but
failed the predeclared frame100 response gate. At frame50/100, position MSE was
`.000001621/.000006153`; y/x R2 was `-2.205/-.139` and z/x
`-1.362/-.228`. Independent strain-versus-frame fits with intercept gave R2
`-.091/-.149` through frame50 and `-.135/-.205` through frame100, so thermal
noise reduction did not rescue response ranking. In comparison, the earlier
v2 bond/contact dynamics remain the best short-horizon compromise: frame50 MSE
`.000000586/.000000556`, y/x R2 `.394/.353`, z/x `-.731/-.690`; frame100 MSE
`.000003509/.000003147`, y/x `.317/.199`, z/x `-1.318/-.989`. Thus coordinate
rollout is good by frames50--100 and one transverse direction has modest
signal, but no model predicts both p-ratio components reliably. Frozen V7
protocol/runner/model SHA256 are
`26dd4681adaead76cfaebd8d49eab249508a304dc1056c06143f67f79105826d`,
`c5885eb7bffc94cd065c3e9125f9bf59881f5b5581b630f9c8ddbfdd751451da`
and `4c073f8e9b17d0b8361691ad82871370241c4dfb922a02f8d8052a5aa9eb4f7d`.

The executed validation notebook is
`notebooks/dynamics/cold_graphs/02_short_horizon_validation.ipynb`:9 cells,
4 code cells, zero errors. It shows rollout MSE and spatial p-ratio R2 versus
horizon, frame100 scatter, and the time-fitted frame50/100 diagnostic. Notebook
SHA256 is
`fd2e2b99c9b89cc2a46e0bc7c435e9e8cc594968d584d5653bfe8fc75f522d40`.
Exact summaries, per-network/per-frame CSVs, checkpoints, recipes, scheduler
ledgers and diagnostic JSONs are under
`notebooks/results/graphs_cold_global_v4/` through
`notebooks/results/graphs_cold_global_v8/`.

V8 returned to the faithful node-only formulation requested by the user. The
3D local--global--local GNN observes frame0/reference positions and frame1
velocity, predicts only node acceleration, and has no box decoder, box target
or box loss. Zero model output is exactly constant-velocity motion. Dynamic
bond/contact edges use prescribed x and fixed reference y/z solely for
minimum-image geometry; y/x p-ratio is computed afterward from affine fits of
rolled-out node positions. The first smoke4690131 exited1 on a CLI variable
typo before loading data or training. Corrected smoke4690133 and production
4690134 exited0 without changing the scientific recipe. Production used
6.19GB and28:06. Phase1 selected epoch9 with one-step position MSE
`1.52e-9`; 20-step training selected epoch2 and stopped after epoch6. At
frames5/10/20/50/100, position MSE was respectively
`8.78e-9/4.27e-8/2.13e-7/1.65e-6/6.42e-6`, while y/x p-ratio R2 was
`-2.909/-1.535/-1.931/-1.603/-2.015` (12/12 valid each). Thus the faithful
global-context simulator rolls coordinates accurately but does not recover
network-to-network transverse response. Phase2 normalized position loss was
highly variable, so a phase1-only v9 recovery harness was prepared, but it was
deliberately not submitted when work returned to the original 200/1,000-MD
study. V8 frozen protocol/runner/model/global-context SHA256 are
`f0295c6b1d873e63174f142f9efcd774471849c5df9ac578144f5aecca9109e2`,
`fee0bc73860d40fffd263000b8cdac1c4564290cfd3f0307b3363a7dcbd8ea77`,
`1c264395a482db08a574393310536df9b88da802749bc170484fa342a29d90e1`
and `7e5fd5617530473571bbd65ed1ad9603a641cf41362c74b2df6a99dceb72ed9f`.
### Declared modest early-frame weighting on 200-MD-step dynamics — 2026-09-18

User hypothesizes that uniform per-frame coordinate MSE still lets the larger
late-deformation residuals dominate the 500-frame AE objective, weakening the
beginning of autonomous rollouts. A matched single-seed pilot changes only the
post-warmup AE frame weights for the completed `joint_low` recipe: frames0..99
receive four times the raw weight of frames100..499, then all500 weights are
normalized to mean one. The same weighted coordinate objective selects the AE
checkpoint. This deliberately avoids the historical inverse-energy schedule's
roughly100-fold early amplification. The fresh2D message-passing/reference16
AE,20node-low+20stiffness-low train IDs0..19/source, validation20..39,
seed123,500-frame horizon, optimizer/budget and frozen one-step499-transition
delta propagator are otherwise unchanged. P-ratio, strain and temperature are
absent from inputs, losses and checkpoint selection. Heldout-development
IDs40..119 are opened only after both checkpoints freeze; report AE and
autonomous rollout p-ratio R2 separately by source/horizon with valid/total.
This pilot tests the weighting hypothesis but cannot establish seed stability.
Protocol `hpc/experiments/mini_temperature200_early_weight/protocol.json` SHA256
`0c016588de11c37922c216e784292289ca79bd4353e77fb8b28c2c105dce939b`;
runner SHA256
`6ab0bc19e36cbfd0e4eb7d02ec3c4c37ba0575fdededc0a69b43899b0ef325ce`.
An end-to-end smoke gates production and frozen evaluation; exact job IDs and
the immutable code manifest will be recorded after submission.

Smoke4690288, production fit4690289 and frozen collector4690290 all completed
exit0 on `mendels_q`,8CPU,16GB,shared placement. Production used4:54:04 and
4.68GB; collection used0:51:44 and4.13GB. Code-manifest SHA256 is
`b457e8d030a01c83fd0eea149d7f93b6680f06bc03ad77c5fac1abb56afa7928`.
The weighted AE selected epoch3 and stopped after epoch15; its frozen one-step
propagator selected epoch9 and stopped after epoch21. AE and propagator SHA256
are respectively
`77dff93b61c2af8aca5fdc8b371725c98f87a8ba6bfb050476e7ecb2370bfd80`
and
`4245563df64040a5b39e67fb651b39b4ec9daee2ca8bf7b8d49c9935bd9e9d70`.

Matched uniform->early-weighted autonomous rollout p-ratio R2 for node-low at
frames25/50/100/499 is `.7362/.8585/.9492/.9573` ->
`.7336/.8637/.9238/.9500`; there is no consistent improvement. For
stiffness-low it is `.0645/.5994/.7495/.8019` ->
`.7379/.7899/.8252/.8397`, a large early and persistent improvement. All
entries are80/80 valid. At frame499 mixed-T same-frame endpoint rollout is
node `.7103`->`.7091` and stiffness `.3390`->`.3843`,80/80 each. The
predeclared frames150..499 mixed-T fit is node `.6124`->`.6214` and stiffness
`.3739`->`.3843`,80/80. AE reconstruction was also evaluated separately:
node-low frames25/50/100/499 `.6748/.8824/.9737/.9547` ->
`.7887/.9449/.9694/.9560`; stiffness-low `.2150/.8416/.9555/.9351` ->
`.7909/.9297/.9669/.9330`, all80/80. Thus modest early weighting specifically
helps the stiffness family and does not justify replacing uniform weighting
globally. This is one matched seed, not a seed-stability claim.

The executed 200-MD notebook02 now includes the baseline four-source rollout
R2-versus-horizon plot/table and a direct uniform-versus-weighted comparison;
all15cells completed with zero errors. Notebook SHA256 is
`c36698cd3727bedc39a96a67e862ec8095eb7275118908bb28a2b8b9e4c8dd5d`.

### Frozen 200-MD latent–boundary correlation analysis — 2026-09-18

The executed 200-MD notebook03 now mirrors the useful low-temperature physical
correlation analysis from the 1,000-MD study without retraining or opening a new
split. It joins the frozen heldout latent paths to the AE-reconstruction rows of
the frozen evaluator by source, trajectory ID and frame. The physical variables
are the evaluator's normalized boundary strains, epsilon_x=deltaLx/Lx(0) and
epsilon_y=deltaLy/Ly(0); p-ratio, strain and temperature remain absent from
training, loss and checkpoint selection. Both all-four seed123 encoders are
shown: one and20 training trajectories/source, heldout-development IDs40..119,
80/80 valid networks/source. Every new trajectory plot is capped at frame200.
The correlation summaries retain frames0..20,0..100 and0..200, covering
0..4,000,0..20,000 and0..40,000MDsteps respectively; frame100 matches the
physical window of frames0..20 in the1,000-MD study.

For the20/source encoder at frames0..100, median within-network absolute
Pearson correlations for node/stiffness are: delta-z0 versus epsilon-y
`.995/.995`, delta-z1 versus epsilon-y `.997/.996`, delta-z0 versus
epsilon-y-minus-epsilon-x `1.000/1.000`, delta-z1 versus
epsilon-y-plus-epsilon-x `1.000/1.000`, and the cumulative deformation-direction
ratio delta-z1/delta-z0 versus minus-epsilon-y/epsilon-x `.992/.993`. Centered
per-step changes q(t+1)-q(t-1), frames1..99, give pooled absolute correlation
`.957/.974` for delta-z1/delta-z0 versus minus-delta-epsilon-y/delta-epsilon-x;
median within-network values are `.892/.898`. Extending through frame200 gives
cumulative direction-ratio medians `.956/.937`; centered frames1..199 give
pooled `.949/.963` and within-network medians `.577/.542`. The pooled ratio
therefore remains strong while longer-window within-trajectory correlations
fall as later local fluctuations accumulate. The component-wise pooled
correlations are coordinate-dependent and weaker, so the ratio is the clearer
cross-family quantity. These are correlations with ground-truth boundary
response, not extra supervision and not rollout metrics.

Frozen input SHA256 values are one/source latent
`2320a3945780799dc60da3d402494903fedeed1cf75599e07c89a25d7dc91261`,
one/source evaluator
`b27d6c85f69a0c201a6fbfb0bf98a00174e4dba08f69441420516be5a207789d`,
20/source latent
`e8f60216fba51ef6767b7345ad5dc7f31b9f6eec8cf8006105d6b1765cf2fd55`,
and20/source evaluator
`0cc648c8a649afebbd21cb1f9d32917439a5ec2dbb74bc70bd6ff353ab7ff6b3`.
Notebook03 executed all9 code cells with zero errors; the final Markdown-only
result update gives notebook SHA256
`d6a412f049981be4e184e4e37bbc2c2faaf78837a4da2998474fe8cfc60c5df2`.

### Declared low-T-only one-network AE at 200-MD cadence — 2026-09-22

The existing one/source200-MD encoder includes node/stiffness mixed-T sources,
so it cannot isolate the paper narrative that one low-T network per family can
organize unseen low-T initial states by final response. A fresh matched
ablation trains exactly node_low ID0 and stiffness_low ID0, with no mixed-T
frames, validates on IDs20..39/source, and opens heldout-development
IDs40..119/source only after each checkpoint freezes. It uses all500 stored
frames0..499 at200MDsteps/frame, the established fresh2D
message-passing/reference-context16 AE, one warmup epoch plus at most60 uniform
all-frame epochs/patience12, and seeds123/456/786. Coordinate reconstruction
validation selects checkpoints. No propagator is trained because the question
concerns initial latent organization only. Temperature, source labels,
p-ratio and strain are absent from inputs, loss and selection.

Post-fit reporting is source-wise initial z0/z1 Pearson and Spearman
correlation with physical frame499 endpoint p-ratio, signed and absolute, with
80/80 valid/total and three-seed mean/SD. Full latent paths for IDs40..59 are
also exported for notebook01. Protocol SHA256
`3eb98128c721ce81ea49f865c1485fe6428c1bc40d0c0c4efbc61b359ed02311`;
runner/PBS SHA256
`9676e51e202b6e1c6ee8eaa460d735938bce33a66e658df85c60e5033d313a49`/
`fad31c3015d959f5b18111decd6c9397b3a43dcf81e1db31c8610994398a9555`.
The new runner imports the previously frozen200-MD code_v6 manifest SHA256
`8ac7a606bb1e88d2332ff3114fe6c0e3eb16552da51453a23d9438c1254dc718`;
unchanged data-manifest SHA256 is
`347c165087b2698131845318ddc37250a9f6689e5f707c2d7b34f2d8fc073f87`.
End-to-end smoke4696553 was submitted to `mendels_q`,8CPU,16GB,
`place=free:shared`,12h, without host pins. It gates production seeds123/456/786
jobs4696557/4696555/4696556 with identical resources. Smoke completed exit0 in
8m36s using4.0GB; checkpoint SHA256
`c8d3a45872bd05c947ddac01e0e125c634a10c15c824c1fcb2334b652d36fab1`.
All three production jobs completed exit0 on `mendels_q` with the declared
8CPU,16GB and `place=free:shared` resources. Jobs4696557/4696555/4696556
(seeds123/456/786) used respectively55m32s/41m59s/48m14s and
4,237,584/4,384,376/4,138,780KiB. Checkpoint SHA256 values are respectively
`ceab58b8f6eab5f44f39142759b1f678d6ab6f5df3a7d3465a3893ac018cf81a`,
`9798763fcc2881733a087176f46aece959db890e5ff702f17bebb5d00c6e2afd`,
and `9016893b45871b85afc15c03e459981c97a8ecf37dfbdf409d8eb625eef354b8`.

For node-low, the strongest absolute raw-coordinate Pearson correlations by
seed are `.891/.961/.165`, giving mean `.673` and sample SD `.441`; strongest
absolute Spearman values are `.884/.945/.121`, mean `.650`, SD `.459`. For
stiffness-low, Pearson values are `.860/.913/.911`, mean `.895`, SD `.030`, and
Spearman values are `.782/.824/.829`, mean `.812`, SD `.026`. Every entry is
80/80 valid. Thus the one-network initial-latent ordering is strong and stable
for stiffness-low, but not seed-stable for node-low: two seeds organize node
response strongly while seed786 does not. Because independently trained raw
latent axes may rotate, the predeclared statistic takes the stronger absolute
z0/z1 correlation within each seed; this is an association diagnostic, not
response supervision or an autonomous-rollout claim.

The strict low-T notebook01 uses seed456 for the displayed trajectories and
shows all three frozen seeds in its summary. All7 code cells executed with zero
errors before the final plot-parity rewrite below.

### 200-MD notebook02/03 plot-parity rebuild — 2026-09-22

At user request, the 200-MD notebooks now retain their cadence-specific plots
and also reproduce the corresponding plot families and ordering from the
1,000-MD notebooks using frozen200-MD artifacts. Notebook02 adds initial latent
state, three-frame-smoothed mixed-T paths, matched one-versus20 low-T paths,
shared cross-family response, early latent/physical response and normalized
shape, response-free latent dimensionality, all-four rollout/final scatter, and
joint-mixed-only rollout. Raw deltaL was not exported at200-MD cadence, so the
physical panels explicitly use the frozen normalized boundary strains
epsilon_x=deltaLx/Lx(0) and epsilon_y=deltaLy/Ly(0). The 200-MD mixed-only run
lacks the historical fitted-prefix artifact, so its panel uses the available
direct framewise physical p-ratio R2 and is labelled accordingly; no value was
invented.

Notebook03 adds the initial z0-z1 endpoint-p plane, static frame0 descriptor
heatmaps, global endpoint versus within-frames0..5 absolute correlations, the
literal two-panel low-T edge-importance comparison, and matched-ID40 early
response. Static descriptors were derived once from all120 frozen frame0
stiffness-low graphs and are thereafter required as a concrete read-only CSV;
descriptor SHA256 is
`0ce7bcec5c2b73d31f6f77cb179fec3222426ecd88fb522fd353656e2e054ce6`.
No model fitting, checkpoint selection, split, or response definition changed.
At the user's clarification, the extra 200-MD-only sections were removed: the
200-MD notebooks now follow the same section and plot sequence as notebooks
01--03 at1,000-MD cadence. Notebook01 remains the explicitly requested
low-T-only exception and retains its three-seed summary; seed456 was selected
post hoc for display because it has strong ordering in both families, while
all three seeds remain visible. A plain raw z0(frame)/z1(frame) panel for five
deterministically sampled heldout networks from each low-T family, restricted
to frames0..99, was also added to both notebook01 versions. The
seed456 training-marker collector and derived CSV SHA256 values are
`eb874de4ea415fec829124f273d60cb0d4d9e73af1d996930e37ae9d1db11166`
and `064ce29f1c718d879cb7bf31db9660bf400e3c5aabe8a8e2305b93ac3f6222fc`.

The notebook-parity rewrite script SHA256 is
`d7644a3b765395bc9d786a81bbcc9f3990071326a27c1db390469985778b9042`.
The executed 1,000-MD notebook01 and 200-MD notebooks01/02/03 have respectively
9/9/10/7 executed code cells and zero error outputs. SHA256 values are
`81c74fd0ab2322ed0be1a5de5a5bb147838f78b1efc6893fa80de639a975fb52`,
`f10ab60665c88de250974997039e7ce8ee5c569d7011de6950e1fac7d8b019c5`,
`25ae749062a37b73c640311cd6a3a2d7fd2ed60e13f1bd070e8669e0e0ceecd4`,
and `28497dafb9d9394e4587bd61e354f8b6805003d0cf22d8b90e925fd8503acc6a`.

### Declared one-step-displacement AE study at 200-MD cadence — 2026-09-22

This matched low-temperature study asks how the 2D latent representation changes
when the decoder predicts the normalized stored-frame increment
`x[t+1]-x[t]` instead of reconstructing displacement `x[t]-x[0]`. The encoder
still receives the current normalized displacement and the same graph/reference
features. It compares 1 and 20 training networks per source for node-low and
stiffness-low, with validation IDs 20..39 and heldout-development IDs 40..119.
Seeds 123/456/786 are predeclared. The architecture, optimizer, uniform
transition weighting, 60-epoch maximum, patience 12 and source-wise validation
rule are fixed. P-ratio, strain, temperature and source labels are absent from
inputs, losses and checkpoint selection.

Frozen evaluation reports one-step position-reconstruction endpoint p-ratio R2
and normalized increment MSE at target frames 1/5/20/100/499, source-wise
with valid/total. Display seed 456 additionally exports all 500 latent frames
for fixed heldout IDs 40..59. Protocol/runner/PBS SHA256 values are
`c8a340dfdc544b89100455040d66d4d55e66dfdbead63d26c8fea361e84da1d1`,
`82d938f4e671a087becca6a14082210fbcf001506b319d908e2f84be10ebcaa6`,
and `4066a9d1676db5441c04fdcafa6f77039d65f6ad9ce2712e3fe75b66eee109bd`.
The frozen base/data manifests retain SHA256
`8ac7a606bb1e88d2332ff3114fe6c0e3eb16552da51453a23d9438c1254dc718`
and `347c165087b2698131845318ddc37250a9f6689e5f707c2d7b34f2d8fc073f87`.
End-to-end smoke 4696712 completed exit 0 in 2m19s using 3,570,848 KiB;
checkpoint SHA256 is
`6eace46d5d5f82f7f066392f4869ac185c9a46e81a131e55455929905d370f6d`.
Production count 1 seeds 123/456/786 are jobs 4696722/4696721/4696720 and
count 20 seeds 123/456/786 are jobs 4696723/4696718/4696719. All use
`mendels_q`, 8 CPU, 16 GB, `place=free:shared`, no host pins, and are gated on
the successful smoke. All six completed; checkpoint SHA256 in the same order:
`ffb38037c17d49e65ad462c60b354578b4e83e26a3a1c8d9ee708fa040b67f91`,
`51044611e2910f8aaa0c57034a252679dfb96736fe6e9b11a996071909834d1e`,
`0c297f56c37ac0f576d7010dd62759f580c7650b1689a4de5734f844f1ddff08`,
`9bd8a78a401e39c57187f69ca9370a198fecb91cd185ff4af0e5637849628bc6`,
`d62bc26a730d1b046b47446e2a28908e37dcc66f1dc4de2fdda3426d2137b4ec`,
`36e42b45822e0d0e28e2e78eebb9a0b79ed0cb2430793bb83ccd41fa37ea68e7`.

Frozen heldout-development results below are mean ± SD over three seeds; each
source/seed/horizon has 80/80 valid networks (240/240 pooled). Values are
teacher-forced one-step endpoint p-ratio R2 and normalized increment MSE,
**not** post-fit p-ratio R2 or autonomous rollout R2. The true current position
is supplied at every prediction, so late-frame endpoint R2 near one must not
be interpreted as long-horizon forecasting.

| Train/source | Source | Target frame | Endpoint p-ratio R2 | Increment MSE |
|---:|---|---:|---:|---:|
| 1 | node-low | 1 | −244.307 ± 8.694 | 1.741 ± 0.068 |
| 1 | node-low | 5 | −2.869 ± 0.472 | 1.194 ± 0.049 |
| 1 | node-low | 20 | 0.993 ± 0.001 | 0.881 ± 0.017 |
| 1 | node-low | 100 | 1.000 ± 0.000 | 1.244 ± 0.027 |
| 1 | node-low | 499 | 1.000 ± 0.000 | 1.380 ± 0.026 |
| 1 | stiffness-low | 1 | −69.225 ± 8.685 | 0.836 ± 0.139 |
| 1 | stiffness-low | 5 | −0.022 ± 0.288 | 0.544 ± 0.111 |
| 1 | stiffness-low | 20 | 0.993 ± 0.002 | 0.471 ± 0.064 |
| 1 | stiffness-low | 100 | 1.000 ± 0.000 | 0.629 ± 0.050 |
| 1 | stiffness-low | 499 | 1.000 ± 0.000 | 0.892 ± 0.030 |
| 20 | node-low | 1 | −94.467 ± 36.530 | 0.408 ± 0.146 |
| 20 | node-low | 5 | 0.070 ± 0.570 | 0.191 ± 0.070 |
| 20 | node-low | 20 | 0.999 ± 0.000 | 0.174 ± 0.033 |
| 20 | node-low | 100 | 1.000 ± 0.000 | 0.217 ± 0.010 |
| 20 | node-low | 499 | 1.000 ± 0.000 | 0.276 ± 0.022 |
| 20 | stiffness-low | 1 | −12.164 ± 3.635 | 0.099 ± 0.023 |
| 20 | stiffness-low | 5 | 0.910 ± 0.019 | 0.046 ± 0.009 |
| 20 | stiffness-low | 20 | 0.999 ± 0.000 | 0.099 ± 0.011 |
| 20 | stiffness-low | 100 | 1.000 ± 0.000 | 0.152 ± 0.018 |
| 20 | stiffness-low | 499 | 1.000 ± 0.000 | 0.277 ± 0.004 |

The [one-step AE notebook](../dynamics/200_md_step/04_one_step_displacement_ae.ipynb)
uses all three seeds for initial-latent correlations and display seed 456 for
matched raw latent paths (fixed IDs 40, 42, ..., 58, including a shared-axis
one-versus-twenty view) and slope associations. Exact unrounded metrics and
valid/total counts are in each
`notebooks/results/mini_temperature200_one_step_ae/count{1,20}_seed{123,456,786}/summary.csv`;
the matching `completed.json` files hold checkpoint hashes and resource usage.
No final-test IDs were used.

### Declared noisy-LJ latent-influence edge pilot — 2026-09-22

Follow-up sparse-edge test (2026-09-23): keep the successful frozen seed123
AE/latent simulator, centered perturbations±0.001 and horizons5/10/20.
Rank the original top-four symmetric-union candidates by the unchanged score;
keep only ceil(10% of candidates) per graph, with canonical-pair tie breaking.
No per-node minimum, existing edges excluded. Train seed123 sparse-latent and
equal-count random GNNs with the original60/20 split/normalization/recipe;
reuse old seed123 base/dense-latent results. Post-fit strain-time-OLS p-ratio
R² at50/100 is primary; other horizons and MSE are diagnostics. Selection uses
no future response or p-ratio. This tests sparsity, not causal edge importance.
New code/artifacts: `hpc/experiments/lj_sparse_influence/` and
`notebooks/results/lj_sparse_influence/`. No notebook or old-result edits.
Frozen `code_v1` jobs: sparse-latent4698612 and random4698610, both completed
training and post-fit evaluation (completion artifacts verified). Live-path4698607 was cancelled;
first frozen attempt4698609 failed on its leftover output directory. Partial
artifacts/logs preserved under `runs/aborted_before_snapshot/`; replacement
uses the unchanged snapshot. Validation edge budget averages60.05 versus596.15.
Outcome: sparsifying to the strongest10% did not preserve the benefit. Noisy-LJ
autonomous post-fit p-ratio R² at frames50/100: sparse-latent -0.746/-2.154,
equal-count random 0.002/-1.024; frozen seed123 base 0.660/0.454 and original
dense-latent 0.700/0.521. All20/20 valid at each horizon, one seed (SD unavailable),
60 train/20 validation, no final test. Sparse-latent coordinate MSE50/100:
3.450e-5/2.407e-4; random4.020e-5/2.810e-4. This rejects this particular
global score cutoff, not sparse augmentation generally; high shared-response
scores alone do not establish critical edges.

This response-blind pilot tests whether a frozen latent simulator can discover
useful nonlocal edges for a physical-node dynamics GNN. The accepted simulator
has one global 2D latent rather than one latent per node, so the study does not
claim a literal node-latent Jacobian. Instead, it perturbs each normalized
global latent coordinate by plus/minus0.001, propagates through the frozen
history-attention/mean seed123 simulator, decodes physical-node positions at
horizons5/10/20, and joins nodes with strong common participation in those
responses. The symmetric union of each node's top-four nonphysical partners is
frozen before downstream GNN fitting.

Three seed123 arms use exactly the same normalized acceleration GNN and noisy-LJ
60-train/20-validation split: the mandatory graph-distance-two/three base graph,
that graph plus size-matched deterministic random nonedges, and that graph plus
latent-influence nonedges. Each GNN has four residual message-passing blocks,
training-only input/target scaling, and LayerNorm in both encoders and every
node/edge block. It observes frames0..3 and rolls autonomously through frame100.
Validation acceleration MSE selects checkpoints; p-ratio, strain, and response
labels do not enter training or selection. Frozen reporting is post-fit physical
endpoint p-ratio R2 at frames50/100 with valid/total, plus physical-coordinate
MSE. No final-test data are used. The frozen AE/cache and latent-propagator
SHA256 values are respectively
`756486f394798a6d51cfef3427a5a2a8845bcc9446c70f4f1c37e805af2619b9`,
`7b0becdc13fcac093034518ff74cf61168e64ff4b4dfc481b27642c528253fa9`,
and `924708b8758c22afea5de8b2124d990ca51658a7419aea52f93a3d9977da61b8`.
The normalized GNN passed three focused tests, including node-permutation
equivariance and LayerNorm coverage. A local two-validation-network end-to-end
functional check completed; its response numbers are not scientific results.
Frozen protocol/runner/PBS/GNN-source/manifest SHA256 values are
`8d122bb76949c285c1412ab192d48a2bb08a774f732e2fb51a7cf7e9205693b9`,
`4f06e069deda3e5d4f28c32544be54173940458d966beb177733652b2350bf6f`,
`f8d8adf747ee6710926060ee6c3605c197b7cc4f6e9a811cdcd874e5a2be9186`,
`a99332ca3507e4f317c4dd8abd53183c1ecfbe9b6a606845cd5ad0ab915cb492`,
and `f1bb046d3704c1519fd86a4a21fd3763485ade5396a8b00129fad02a941770bd`.
All three end-to-end smoke jobs completed exit0: base4696918,
random4696920, and latent4696921. The two-network smoke response values are
only plumbing checks. Production base/random/latent jobs are respectively
4696922/4696924/4696923 on `mendels_q`,8CPU,16GB,
`place=free:shared`, with no host pin. All three completed exit0 in
1m58s/2m46s/2m37s using2.63/3.08/3.03GB. Their checkpoint SHA256 values are
`b2d4097f464dfc3ca2504e85efe25947fb967aa8be573e3bc0e5e889ca2c3327`,
`ba29b8afaec21a237a1d04d7039ea8e9f169eb44813ed0bb759d1cea432a6e24`,
and `e4985c9856e2192f86b68c903bc95c42c7e87628bb193d65e32ed9bd2df6997d`.

For seed123, the primary post-fit p-ratio R2 at frames50/100 is
`.660/.454` for base, `.084/-.287` for size-matched random, and `.700/.521`
for latent-influence edges, with20/20 valid in every cell. Endpoint R2 gives
the same ordering: `.681/.455`, `.100/-.151`, and `.718/.530`. Coordinate MSE
at frames50/100 is base `3.4e-5/2.29e-4`, random `3.0e-5/1.99e-4`, and latent
`3.1e-5/2.10e-4`; random edges therefore have the best coordinate MSE while
the worst response R2. The latent improvement over base is promising but small,
and a paired20-network bootstrap of response squared-error improvement crosses
zero at both horizons. It is not yet a robust success claim.

The exact post-fit evaluator SHA256 is
`a05df7a7ee678a69a9ac653a76961f531a91571f8eb43ab7090402031fb8875f`.
It uses two physical-strain OLS fits with intercept over frames0..horizon and
takes `-slope_y/slope_x`; it is applied only after frozen checkpoint selection.
Validation latent edges are longer than random controls on average
(`1.344` versus `1.134` in normalized reference coordinates), and their mean
per-network Jaccard overlap is only `.034`. To test initialization robustness
without changing the seed123-discovered graph, matched downstream seeds456/786
were submitted for every arm as jobs4696939/4696941 (base),
4696938/4696936 (random), and4696937/4696940 (latent). These six jobs use the
same declared resources. All completed exit0. Their walltimes were respectively
1m44s/2m18s,3m22s/2m14s, and2m47s/2m11s, with peak memory2.39--3.11GB.
The seed456/786 checkpoint SHA256 values are base
`a45cec14cabd05d9055fcdbbc8624185a571025dc33bb61fc603a47ac253c27a`/
`c3e41debc2e039ad7e01fa50f9b7bbd3e43a3f053945fe5d6a118ba1d3dc45af`,
random
`cd9f073e4362ebf2b9dceae60a0252b1e8bc5321808f97eca5153fd273bd3592`/
`a7521883e975bff05b9a65d5a2b662482d59ff8db57169ffdb325e317bf783fd`,
and latent
`0b38196cb0471948b531bca9a5554b25f22f8b5f8d9410e6a535337c4f037c40`/
`3c94a92a29b42c5b685f35684506f04fd61757f096939fcc7d45249e0c8a9602`.

Across downstream seeds123/456/786, post-fit p-ratio R2 mean plus/minus sample
SD at frame50 is base `.652+-.053`, random `.425+-.296`, and latent
`.696+-.012`; at frame100 it is base `.397+-.196`, random `.096+-.354`, and
latent `.497+-.027`. Every cell is60/60 valid. Coordinate MSE mean at
frames50/100 is base `3.354e-5/2.270e-4`, random `3.312e-5/2.226e-4`, and
latent `3.135e-5/2.103e-4`. Thus this pilot supports the hypothesis: the
latent-influence graph improves mean response R2 by `.044` at50 and `.099`
at100 relative to the base while sharply reducing seed variability, and it
also beats size-matched random edges. This remains one fixed edge-discovery
simulator seed and one20-network validation split, not a final generalization
claim.

The exact code_v2 protocol/runner/PBS/manifest SHA256 values are
`0584800560dfe795f65569766382d82b476296479a0c9c1d6305db0ebd4988a9`,
`5d86b0319f5c46195c90b2be64ee7ad0c3a7538ba1f2be255b3af13e433a3b18`,
`221189fa6b794369d932e5c941c1540e51a22a5afcbf3ab8fc1b24b8a8ad1d2f`,
and `c39515ba13e6b090d35c759c1921b040b675b59f158314a74384978a690a18db`.
The final post-fit evaluator/seed-summary/per-network/edge-list SHA256 values are
`8fe3196e9e1c826bc5f660e0c3a164c654447fbc33910141de6d06f33566f76b`,
`05d11a2efe89ff89e293fff814bb2151d27d2b6dfb9ac65795fa137f31fc130e`,
`a73bc921d500fcacecb596302f1d239cb2646a507f7d8e59fa550e7fea20fa22`,
and `a7efc07139c3d0f265a8dd00483a355ad787b43605cfe8dd36e444492f70db1a`.

### Declared LJ-only latent-influence edge replication — 2026-09-22

At user request, this replication removes Reid/low-temperature exposure from
the edge-discovery representation. It reuses the exact existing
LJ-only seed123 4D attention-reference16 AE checkpoint rather than duplicating
its3h08m fit. That AE was trained on60 noisy-LJ trajectories x200 frames and
validated on20 noisy-LJ trajectories; it contains no Reid/de Pablo examples and
used no response supervision. Checkpoint SHA256 is
`42d36dc32c6a4bee95f242e7f13f8be6947fb1a49870d26cdf00a9417395164a`.

A fresh seed123 history-attention/mean latent propagator will fit frames0..100
of the same60 training trajectories and use decoded coordinate validation MSE
on the same20 validation identities for checkpoint selection. After freezing,
centered latent perturbations plus/minus.001 at horizons5/10/20 will select the
symmetric union of each node's top-five nonphysical partners. On heldout graphs,
edge discovery may read only the frame0 reference graph and z0; validation
future positions and p-ratio are forbidden. The mandatory graph-distance2/3 LJ
relations remain in every arm.

Downstream base/random/latent acceleration GNNs retain the prior architecture,
normalization,60-train/20-validation split, frames0..3 initialization,
frames4..100 autonomous rollout, seeds123/456/786, response-blind checkpointing,
and post-fit p-ratio R2 at50/100. A new notebook will include seed summaries,
scatter plots, and an unseen validation graph with physical and selected latent
edges visually distinguished. No final test data are used.

Frozen protocol/preparation/PBS/downstream/PBS/manifest SHA256 values are
`6fe344e99a210c816ad654c9c8499c9926610dc27a0059a3f5000f4bd1604e4c`,
`b2b0975fcbf22dba1b1794049bbc20cfa6c8dcaaaf43ada08a0b1fba38252490`,
`821cdae54ba20eb48df39cc8c73c10dd77ffcadee8e5ea31d9907d980b08b2c3`,
`b7096d6f92dbe1dd869cd4cbcd4c5a51893f176039f5814215a3301b2f1a55d7`,
`9191f94d035cc3799fbac26abfa3031f9d8c66d28fe73050a88dd45df6ed8f2f`,
and `9b69006092110bed1230518a4c8d806ef48c453c3971957e28d71b208ed0b400`.
The four-train/two-validation end-to-end latent smoke job4696979 completed
exit0 and preserved the AE hash; its response-free numerical values are plumbing
only. Full60/20 latent-cache and propagator job4696984 was submitted to
`mendels_q`,8CPU,16GB, `place=free:shared`, with no host pin. Job4696984
completed exit0: encoding plus fitting used the declared60/20 split, stopped
after15 epochs with best response-blind decoded validation MSE `.0410562`, and
preserved the frozen AE hash. Cache and latent-model SHA256 values are
`f5e1f378e5fef7478a0fdd5a6fd7f458cbc05f271a931d6880482514db87d1ec`
and `ae5f2b145d46ef1e6bf47d520fd0f2e9547fff016e793046f3bb1f23f437d904`.
The downstream latent-arm smoke job4696992 completed exit0; its two-network
response values are plumbing only. Top-five selection added an average832.5
unique undirected edges on those two smoke graphs. The physical base arm is
mathematically unchanged, so its existing seed123/456/786 frozen checkpoints
are reused rather than retrained. New top-five random jobs are
4696997/4696994/4696999 and LJ-only latent-edge jobs are
4696996/4696998/4696995 for seeds123/456/786. All use the declared shared PBS
resources and completed exit0. Across the20 unseen validation graphs, Top5
selection adds741.75 unique undirected edges/graph on average (physical plus
distance2/3 base mean1997.45). Primary post-fit p-ratio R2 mean plus/minus
sample SD at frame50 is base `.652+-.053`, random `.547+-.175`, and LJ-only
latent `.516+-.283`; at frame100 it is `.397+-.196`, `.061+-.610`, and
`.291+-.289`. Every cell is60/60 valid. Coordinate MSE at50/100 is base
`3.354e-5/2.270e-4`, random `3.428e-5/2.443e-4`, and latent
`3.273e-5/2.228e-4`. Thus this LJ-only4D-attention replication does not beat
the base response result and does not reproduce the earlier shared-MP2D latent
edge gain. Architecture, latent dimension, normalization and AE exposure differ,
so this contrast cannot establish that cross-family exposure is causal.

Post-fit evaluator SHA256 is
`5a316127fb501f0dcc365f1a440a1fc08ff909614991f88be4307c373827716e`.
Executed notebook `notebooks/dynamics/11_ljonly_latent_influence_edges.ipynb`
(SHA256 `1866748ef47d497f54ba1a08da6cd19d856b0516f95d4ea1db35ea0680d0a5cc`)
contains the three-seed response curve, frame100 scatter and an unseen graph
visualization with all selected edges plus the60 strongest highlighted. It has
three executed code cells and zero errors; no figure files were exported.

### Declared matched noisy-LJ AE-scope edge study — 2026-09-22

The prior shared-MP2D versus LJ-only-attention observation is confounded and is
not used to claim that other source families improve the noisy-LJ AE.  At the
user's request, the new `lj_matched_scope` study changes only the composition of
four training trajectories: `ljonly4` sees four noisy-LJ graphs, while
`shared4` sees one Reid, one low-T, one mixed-T and one noisy-LJ graph.  Both
use the same 2D attention AE, hidden size128,32 tokens, five edge channels,
frames0..100,12120 sampled frame rows per epoch,40-epoch/patience8 budget, and
the same20 unseen noisy-LJ validation graphs for response-blind coordinate
checkpointing.  All LJ stages retain physical plus graph-distance-two/three
relations.  Seeds123/456/786 are paired.  No p-ratio, strain, source label or
final-test result enters fitting or selection.

Code_v1 smoke jobs4697019 (`ljonly4`) and4697018 (`shared4`) exited1 in2s
before experiment import because the PBS launcher selected the repository venv,
which lacks torch; dependent4697021--4697026 did not run.  This is a launcher
failure, not a model result.  Code_v2 changes only the interpreter to the
established torch environment.  Replacement smokes are4697032/4697031 and
production jobs, held `afterok` both, are ljonly4 seeds123/456/786 =
4697035/4697034/4697037 and shared4 =4697038/4697039/4697036.  Every job uses
`mendels_q`,8CPU,16GB, `place=free:shared`, no host pin.  Frozen code is under
`notebooks/results/lj_matched_scope/code_v2`; jobs and dependencies are in
`jobs.json`.  After AE completion, each frozen AE will receive the same
response-blind latent-dynamics/Top5 influence-edge and normalized-acceleration
GNN evaluation on unseen noisy-LJ graphs.  Frozen stage-two code_v3 manifest is
`6d4824fe63b41524a1e4ab7b62977667b4b723789c35a06b53ae8d8bdbc04b36`.
Latent smokes4697045/4697046 gate paired full propagators: ljonly4
4697050/4697051/4697049 and shared4 4697048/4697052/4697053.  Downstream smokes
4697055/4697054 gate paired latent-edge GNN jobs: ljonly4
4697058/4697057/4697059 and shared4 4697056/4697061/4697060.  Dependencies and
exact mappings are in `jobs.json`. Results are pending.

The first code_v3 latent smoke4697045 failed before fitting because the inherited
propagator assumed16D reference embeddings plus2D positions, while the matched
attention AE supplies128D reference embeddings plus2D positions.  Its downstream
dependency chain4697046/4697048--4697061 produced no scientific result.  Code_v4
changes only the propagator token-width constructor to read the frozen AE output;
its manifest SHA256 is
`7d3a5f04c244983877a06b7fcdc79cb48ccda5e419862af8e2d21318e948842a`.
Replacement latent smokes are4697075/4697074; full latent jobs are ljonly4
4697082/4697076/4697079 and shared4 4697080/4697077/4697078.  Replacement GNN
smokes are4697087/4697086 and full GNN jobs are ljonly4
4697088/4697093/4697089 and shared4 4697091/4697090/4697092.  The final notebook
will live in the dedicated `notebooks/lj_matched_scope/` folder and report
post-fit p-ratio R2 every10 frames from10 through100, with three-seed SD and
valid/total counts.
### Declared matched MP2D cross-family regularization study — 2026-09-22

The user rejected the preceding four-trajectory screen as too data-poor and
requested at least20 networks/source.  New `lj_mp2_scope20` isolates source
composition at equal total trajectory/frame exposure: `ljonly60` trains on60
noisy-LJ trajectories, while `shared3x20` trains on20 Reid +20 low-temperature
de Pablo +20 noisy-LJ trajectories.  Both use the same2D
message-passing-reference16 AE, two message-passing steps, hidden96,32 latent
tokens, all200 frames,12000 frame samples/epoch,60-epoch/patience12 budget and
seeds123/456/786.  Both select checkpoints using coordinate reconstruction on
the same20 unseen noisy-LJ validation trajectories only.  Graph-distance2/3 LJ
relations and the five-channel edge schema remain consistent.  P-ratio/strain,
source labels and final-test data do not enter fitting or selection.

Smokes4697196/4697197 gate production ljonly60 jobs4697201/4697200/4697203
and shared3x20 jobs4697204/4697199/4697202.  All use `mendels_q`,8CPU,16GB,
`place=free:shared`, no host pin.  Frozen code_v1 manifest SHA256 is
`cc3e1970f2624d9c2fc71c8bf6a4f148934e5e119be6c4cf9697bbaa44cb2874`;
exact mapping is in `notebooks/results/lj_mp2_scope20/jobs.json`.  Post-fit
physical reconstruction p-ratio R2 on the common LJ validation split is the
primary comparison; downstream edge tests are gated on that result.

At 2026-09-22 19:01 IDT, all three `shared3x20` runs and `ljonly60` seed123
had completed successfully; `ljonly60` seeds456/786 remained healthy and
running.  Frozen-checkpoint reconstruction evaluation job4697375 is held
`afterok` those two jobs and will compute the predeclared physical post-fit
p-ratio R2 horizons and coordinate-MSE secondary diagnostic automatically.

All six AE jobs subsequently completed successfully: `ljonly60` seeds
123/456/786 trained for20/60/48 epochs and `shared3x20` seeds123/456/786 for
19/17/19 epochs.  Evaluation job4697375 failed before producing metrics because
the decoded graph intentionally omitted the reference-box metadata needed for
physical-coordinate inversion.  The evaluator now takes this frozen metadata
from the original reference graph; replacement job4697566 was submitted with
unchanged checkpoints, split and metric.

Replacement evaluation4697566 completed successfully in10m51s.  On the common
20-network unseen noisy-LJ validation split, three-seed AE reconstruction
post-fit p-ratio R2 (mean +/- SD; valid/total aggregated across seeds) strongly
favours `shared3x20`: at frame50,0.220+/-0.283 versus
`ljonly60` -1.379+/-1.185 (60/60 each); frame100,0.527+/-0.150 versus
-2.312+/-2.361 (60/60 each); frame199,0.252+/-0.144 versus
-3.544+/-3.148 (60/60 each).  Yet physical coordinate MSE is nearly matched:
at frame100,0.000381 shared versus0.000391 LJ-only; at frame199,0.001921 versus
0.001999.  Thus cross-family exposure improves preservation of the small
mechanical-response signal rather than merely lowering average position error.
These are frozen AE reconstructions, not autonomous latent/GNN rollouts.
Exact per-network, per-seed and aggregate tables are in
`notebooks/results/lj_mp2_scope20/evaluation/`; checkpoint hashes and the
response-blind evaluation contract are in `evaluation.json`.

The downstream latent-influence study then froze a2D history-attention
propagator with mean reference context for each of the six AEs.  Smokes
4697589 (LJ-only) and4697593 (shared) passed.  Initial shared smoke4697588
failed because Reid/de Pablo graphs lack LJ-specific `source_sim_id`; the
bookkeeping was corrected to use exact recipe split indices, without changing
data or learning.  Full latent jobs are LJ-only seeds123/456/786 =
4697600/4697597/4697598 and shared seeds123/456/786 =
4697601/4697599/4697596.  The latent objective and checkpointing remain
response-blind decoded-position/latent losses; final-test data are unused.

The matched downstream protocol uses the exact same canonical60 noisy-LJ
training and20 held-out noisy-LJ validation trajectories for every arm; it
does not incorrectly reduce the shared-AE GNN to its20 LJ AE-training graphs.
For each paired seed, canonical LJ graphs are re-encoded through the matching
frozen AE/latent model.  The physical GNN is unchanged (normalized
acceleration, hidden64, four MP blocks), and compares base, exactly
count-matched random nonedges, and top-five-per-node propagated latent
influence nonedges (epsilon0.001; horizons5/10/20), excluding the existing
physical plus distance-two/three relations.  Smokes4697608/4697607 gate18
production jobs4697610--4697627; exact mapping is in `jobs.json`.  Selection
uses one-step acceleration MSE only, while full physical paths through frame100
are saved for post-fit p-ratio R2 at10/20/30/40/50/75/100.

Operational update: all six latent models completed.  The downstream pilot
found a package import conflict and then a saved-path field error; neither
affected the learned models.  The incomplete runs were cancelled and retained
under `downstream/aborted_before_output_fix`.  Corrected18-job GNN matrix
4697663--4697680 and post-fit evaluator4697681 are running/queued; exact
scope/arm/seed mapping and failure identities are in `jobs.json`.

Final: all18 corrected GNN jobs and post-fit evaluator4697681 finished with
exit0.  On the same20 unseen LJ graphs, frame50/100 autonomous post-fit p-ratio
R2 (three-seed mean +/- SD,60/60 valid per cell) was base
0.679+/-0.037 / 0.444+/-0.083; LJ-only latent edges
0.414+/-0.289 / 0.117+/-0.312; shared-AE latent edges
0.615+/-0.145 / 0.336+/-0.282.  Count-matched random edges were
0.560+/-0.226 / 0.102+/-0.671 for LJ-only and
0.522+/-0.291 / 0.311+/-0.309 for shared.  Shared training improved its
learned-edge arm relative to LJ-only, but neither learned-edge arm beat the
physical baseline.  Mean added undirected edges/network were739.8 and740.9
for LJ-only and shared, respectively.  Full horizon/seed/per-network metrics,
coordinate MSE, valid counts, artifact hashes, and estimator are under
`notebooks/results/lj_mp2_scope20/downstream/evaluation/`.  The executed
`notebooks/lj_mp2_scope20/01_augmented_gnn_comparison.ipynb` shows the curves,
frame100 scatter and edges on held-out network15; eight cells, no errors.
The final-test split was not used.
