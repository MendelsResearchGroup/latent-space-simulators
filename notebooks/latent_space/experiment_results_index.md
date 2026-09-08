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

Architecture research and overnight expansion delegated to Terra (`terra_shared_ae`) at user request. Durable instructions and primary-source hypotheses: `docs/research/shared_dynamics_architecture_brief.md`. Priorities: nonlinear neighbor message passing, pooling/capacity controls, followed conditionally by learned temporal state and rollout robustness. Broad queued AE experiments authorized after smoke checks; submission/results will be recorded by the executing agent.

Deeper architecture audit: `docs/research/shared_dynamics_deeper_audit.md`. Saved history diagnostics suggest LJ training error is also high (~0.30 normalized MSE vs ~0.33 validation); no irreducible-noise claim. Identified nonzero-mean standardized-edge reversal inconsistency in frozen model; algebraic check `scripts/audit_ae_edge_reversal.py`, downstream impact pending. Terra notified to isolate correction from architecture changes. Proposed encoder-vs-decoder latent optimization, latent-use controls, and predictive temporal objectives without expert supervision.

Consolidated execution plan: `docs/research/shared_dynamics_execution_plan.md`, assigned to Terra at user request. Stages A–F cover correctness controls, information-bottleneck diagnostics, broad spatial architectures, temporal representations/history, source interference and rollout robustness. Early independent work can queue after smoke checks; later stages depend on recorded state-based evidence.

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

Pipeline audit saved at docs/research/latent_pipeline_audit.md. Default cache
configuration matching is now enabled and regression-tested; current study
force-training means stale reuse does not explain its outcomes. Targeted active
pipeline suite:17 passed plus6 subtests, including batch-vs-single parity,
normalization inversion and future-data independence at fixed latent. Three
separate legacy simulator tests failed on stale argument signatures; recorded
rather than hidden. Edge-reversal corrected controls and architecture study
remain delegated to Terra, with smoke failures and retries preserved.

User reports working LJ AE elsewhere. Historical08/09 comparison saved in docs/research/lj_historical_positive_control.md:08 LJ R2 .6296 on30 vs .4681 on110;09 .468 on110 under ordinary reconstruction selection. Recent6D differs materially in data budget,width,decoder tokens,objective,edges and evaluation split; it was not a positive-control replay. Exact user example requested; checkpoint not found in current08/09 results directory inventory.

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
`docs/research/compact_reference_lj_next_plan.md`. Plan only, no new submissions.
Prioritize matched shared/LJ-only reconstruction diagnostics with16D reference
and wide control, full-time coverage, then conditional propagator training.

User clarification: next noisy-LJ experiments MUST retain LJ-edge augmentation
through original spring-graph distance3 (new distance2/3 pairs, raw stiffness0,
relation indicator1). Shared five-channel edge schema; consistent train/val/inference.
Plan and AGENTS.md updated. Older four-channel architecture runs lack this
required representation and cannot settle LJ performance with the intended edges.

### Compact-reference LJ reconstruction execution prepared — 2026-09-07

User authorized continued reconstruction work and a motion/data audit. Exact
50-run first matrix and limitations: docs/research/compact_reference_lj_next_plan.md.
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
