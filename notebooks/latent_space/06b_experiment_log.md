# 06b experiment log

## 2026-08-21 — AE all sources; propagator Reid + low-T

- Notebook: `06b_mixed_dataset_shared_latent_rollout.ipynb`
- Seed: `3456456`; 4D latent; 20 train and 20 validation trajectories per source; frames `0..100`.
- AE training sources: Reid, de Pablo low-T, de Pablo mixed-T, noisy LJ. AE stopped at epoch 11 of 14. Validation step-100 p-ratio R² at that epoch: Reid `0.967`, low-T `0.944`, mixed-T `0.776`, noisy LJ `0.663`.
- Propagator: one-step shared Δz MLP, hidden size 64, static 16D mean-pooled reference context, equal per-source loss; training sources only Reid and de Pablo low-T. It stopped at epoch 3 of 6.
- Held-out step-100 rollout R² (30 trajectories/source): Reid `0.319`, low-T `0.943`.
- Propagator-unseen mixed-T step-100 rollout R² (30 held-out trajectories): `0.901`.
- Gradient cosine (one fixed 32-transition probe batch after training): Reid ↔ low-T `-0.149`.

Noisy LJ and mixed-T were excluded from propagator supervision. Mixed-T remained an AE-trained, propagator-zero-shot evaluation source; noisy LJ was AE-only in this run.

## 2026-08-21 — frozen all-source AE; Reid + low-T + noisy-LJ propagator with stride-5 PCGrad

- Notebook: `06b_mixed_dataset_shared_latent_rollout.ipynb`
- Frozen AE: `model_compact_edges_v8_ae_all_sources_prop_reid_lowT_lj.pt` (the all-four-source 4D AE above); no AE retraining in this experiment.
- Propagator: shared one-step Δz MLP (hidden size 64), 16D mean-pooled static reference context, equal source loss, genuinely balanced source-mixed batches, source-specific latent/Δz standardization fitted on propagator-training trajectories only, and PCGrad.
- Training sources: Reid, de Pablo low-T, and noisy LJ; 20 training and 20 validation trajectories/source; frames `0..100`; stride `5`; seed `3456456`.
- Checkpoint selection used the macro source-wise validation rollout R². Best epoch: 5 of 8; validation step-100 R²: Reid `0.720`, low-T `0.897`, noisy LJ `-0.378`.
- Held-out step-100 rollout R² (30 trajectories/source): Reid `0.749`, low-T `0.903`, noisy LJ `-0.032`.
- Propagator-unseen mixed-T held-out step-100 R²: `0.777`.
- Raw source-gradient cosine probes (32 batches): Reid ↔ low-T mean `-0.042`, negative fraction `0.500`; Reid ↔ noisy-LJ mean `0.060`, negative fraction `0.406`; low-T ↔ noisy-LJ mean `-0.117`, negative fraction `0.625`.

PCGrad projected conflicting gradients during training, but it did not make noisy-LJ's shared rollout viable. The remaining failure is source-specific noisy-LJ dynamics, rather than an unbalanced batch or an inactive PCGrad path.

## 4D historical-recipe reconstruction — submitted

The original v8 checkpoint is absent from this workspace. Git history preserves
older 2D and later 32D notebooks, not the exact 4D run. Therefore these jobs are
explicitly recipe reconstructions, not exact replays of the held-out 0.901 score.

Documented core: 4D attention AE trained on all four sources, 20 train/20
validation trajectories/source, frames 0–100; width-64 one-step delta MLP with
16D physical mean reference context, trained and selected on Reid + low-T only.
AE cap 14 epochs and propagator cap 6. Seeds 3456456 (historical), 123, 456.

Reconstruction assumptions: AE width 96/32 tokens from older notebook lineage;
batch 32, AdamW lr 1e-4, weight decay 1e-5, patience 3; compact four-channel
stored edges; no LJ edge augmentation/indicator; current audited per-axis box
normalization; forecast origin 0, endpoint p-ratio; macro-source rollout
checkpoint selection. Use a single fresh AE-plus-propagator call without an
intermediate reseed. These settings are not all recoverable from the old log.

Deliberate split difference: first 20 current matched training IDs and the same
20 validation IDs per source. Explicit empty test split protects the reserved
final-test partition. Report every source separately, including mixed-T and LJ,
at frames 5/10/25/50/75/100. No pooled success criterion.

Frozen source and scripts: `notebooks/results/06b_4d_reconstruction/code_v1/`.
Exact per-run recipes and completion markers are saved under `d4_s<seed>/`.
Job ledger: `notebooks/results/06b_4d_reconstruction/jobs.jsonl`. Jobs use
`mendels_q`, shared placement, eight CPUs, 32 GB, no host pins. Compilation and
PBS syntax checks passed. Results pending. Establish this 4D baseline before
adding LJ supervision or changing latent dimension; no 8D substitution.

## Matched 2D versus 4D expansion

User clarified that 4D is not a requirement: test whether 2D suffices and use
available shared cluster capacity for broader useful comparisons. Expand to
2D/4D × seeds 3456456/123/456/786/2026 (ten conditions). Retain the three
already-submitted 4D runs and submit only seven missing conditions. All source
and training settings remain fixed apart from latent dimension/model seed.
`code_v2` generalizes the runner's dimension argument; its underlying `src/`
SHA-256 hashes were verified identical to `code_v1`. Hash manifests are stored
with each code snapshot. The original-recipe limitations still apply.

All outcomes, including negative results, remain reusable under
`notebooks/results/06b_4d_reconstruction/` (directory name retained for existing
runs). `aggregate_06b_dimension_reconstruction.py` writes completion status,
source-wise raw/aggregate results, and within-seed 2D-minus-4D contrasts.
Do not infer dimensional necessity from a single seed or pooled score. Prefer
2D when source-wise validation supports it; do not use reserved test for choice.
New jobs use `mendels_q`, shared placement, eight CPUs, 32 GB, no host pins.

## Reconstructed 06b dimensionality comparison — ten runs complete

All 2D/4D × five-seed runs completed on the protected matched validation
split. Exact recipes and deliberate historical differences remain as above.
AE sees all four sources; propagator supervision and selection use Reid and
low-T only. Seeds 3456456/123/456/786/2026. All metrics below retain 20/20
validation networks per source per seed. Frame-100 p-ratio R² (mean ± SD):

| Evaluation | Dimension | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| ae | 2 | 0.926 ± 0.045 | 0.966 ± 0.011 | 0.893 ± 0.032 | -0.144 ± 0.651 |
| ae | 4 | 0.938 ± 0.026 | 0.971 ± 0.010 | 0.932 ± 0.026 | -0.641 ± 1.584 |
| rollout | 2 | 0.814 ± 0.044 | 0.947 ± 0.011 | 0.904 ± 0.012 | -10.549 ± 3.505 |
| rollout | 4 | 0.698 ± 0.084 | 0.923 ± 0.036 | 0.877 ± 0.042 | -8.219 ± 2.619 |

Strong mixed-T propagator transfer is recovered across five seeds under this
reconstructed recipe. This is new validation evidence, not an exact replay of
the historical held-out 0.901 result. 4D is not necessary: 2D has better mean
rollout R² for Reid/low-T/mixed-T, despite slightly lower AE reconstruction.
All five 2D mixed-T rollout scores fall between 0.895 and 0.923.

Noisy-LJ remains unresolved. It has no propagator supervision here, and its
AE response reconstruction is unstable already. Do not attribute its entire
rollout failure to the propagator. Five seeds share the same validation
networks; SD measures training randomness, not uncertainty across new datasets.
No initial-coordinate predictive probe was run on these new checkpoints, so
initial-response claims cannot be transferred from earlier low-data AEs.

Adopt 2D as the working baseline for the established three sources. Preserve
these checkpoints and split identities. Next isolate adding LJ supervision
on a frozen AE from repairing LJ representation; keep source scaling and
checkpoint-selection changes explicit. Compare learned motion with the same
causal-prefix baselines. Do not change dimension, AE recipe, dynamics recipe
and source mixture together. Reserve final-test evaluation until selection
is fixed. No new training launched during this synthesis.

## Mixed-T AE exposure ablation — submitted

Question: does the successful 2D reconstruction still transfer to mixed-T when
mixed-T is excluded from AE exposure as well as propagator supervision?
Reuse five completed includes-mixed-T baselines; add five excludes-mixed-T runs
with seeds 3456456/123/456/786/2026. Each new run copies its paired baseline
recipe, removes only mixed-T from the fitting mixture, and changes the output
path/labels. Underlying frozen training/model code hashes match `code_v2` of
`06b_4d_reconstruction`. Retained-source split IDs and every training
configuration value were verified unchanged (apart from cache path).

AE: 2D, width 96, 32 tokens, 20 train/20 validation networks per retained
source (Reid/low-T/LJ), frames 0–100, maximum 14 epochs, patience 3, batch 32,
lr 1e-4, weight decay 1e-5. Propagator: unchanged width-64 delta MLP, mean
physical context 16, only Reid + low-T supervision and selection, max 6 epochs,
patience 3, same optimizer/seed/forecast origin and evaluation horizons.
Raw inputs with audited in-memory normalization; no reserved test data.

The excluded mixed-T source is loaded only after AE and propagator fitting and
checkpoint selection finish. Mixed evaluation has zero training/test rows and
the same 20 validation IDs as the baseline. It cannot affect AE statistics,
weights, or checkpoint selection. Combined CSVs contain all four sources;
`bundle.pt` is saved by the training call before the extra mixed-T evaluation,
so its cached evaluation rows cover only the retained sources. The full mixed-T
evaluation specification is preserved in the run recipe.

Qualification: source removal reduces examples/updates per epoch and changes
AE-fitted normalizers and random-number consumption. Equal model seeds do not
ensure identical propagator initialization after different AE runs. This tests
whether the same exclusion recipe works, not universal necessity or an isolated
mechanism for any failure. All results must remain source-wise with counts.

Results/ledger: `notebooks/results/06b_ae_mixed_ablation/`. Frozen scripts/source
and SHA-256 manifest in `code_v1/`. Collector:
`scripts/aggregate_06b_ae_mixed_ablation.py` reports source-wise AE/rollout scores
and paired excluded-minus-included contrasts. Five new shared-node PBS jobs,
`mendels_q`, eight CPUs, 32 GB, no host pins. Compilation, shell syntax, exact
configuration/split assertions and source-code hash checks passed. Results pending.

## Mixed-T AE exposure ablation — five paired seeds complete

All five exclusions completed. The included conditions reuse the completed
successful 2D runs. Recipe, identical retained-source split IDs, post-training
loading of excluded mixed-T, and sample-budget/RNG qualifications are recorded
above. No mixed-T training, statistics fitting, or checkpoint-selection exposure
occurs in the exclusion condition. Propagator supervision is Reid + low-T in
both. Frame-100 validation p-ratio R² (five-seed mean ± sample SD):

| AE exposure | Evaluation | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| exclude_mixed | ae | 0.903 ± 0.051 | 0.958 ± 0.015 | 0.753 ± 0.102 | -0.011 ± 0.342 |
| exclude_mixed | rollout | 0.802 ± 0.093 | 0.913 ± 0.039 | 0.868 ± 0.033 | -12.195 ± 5.034 |
| include_mixed | ae | 0.926 ± 0.045 | 0.966 ± 0.011 | 0.893 ± 0.032 | -0.144 ± 0.651 |
| include_mixed | rollout | 0.814 ± 0.044 | 0.947 ± 0.011 | 0.904 ± 0.012 | -10.549 ± 3.505 |

All source/horizon/seed rows retain 20/20 validation trajectories. Mixed-T
AE exposure is not necessary for strong endpoint response rollout in this
recipe: excluded-source scores span 0.832–0.909 across the five seeds.
Exclusion-minus-inclusion paired endpoint change is -0.0365 ± 0.0439.
Exposure improves mean endpoint accuracy but does not establish necessity.
Five seeds share the same evaluation networks; this is development validation,
not final held-out-test confirmation or independent dataset replication.

Mixed-T exclusion rollout R² at frames 5/10/25/50/75/100 is
0.071/0.296/0.357/0.547/0.669/0.868, versus
0.079/0.311/0.404/0.574/0.732/0.904 with AE exposure. Early-time response
prediction is substantially weaker than the endpoint. AE reconstruction on
excluded mixed-T averages 0.753 at frame 100, below the rollout score; response
metric differences alone do not prove a learned denoising mechanism.

Noisy-LJ remains poor in both rollout conditions, with no propagator LJ
supervision and unstable AE response reconstruction. Next prioritize matched
simple kinematic/reference-only baselines on the successful 2D checkpoints
and controlled LJ-supervision/representation comparisons; preserve this
fully unseen mixed-T baseline. No additional training launched while collecting
these results. Exact per-source/seed metrics and paired comparisons are under
`06b_ae_mixed_ablation/`.

## Noisy-LJ representation repair before dynamics — AE-only matrix submitted

User required repairing noisy-LJ AE reconstruction before adding LJ propagator
supervision. The successful mixed-T-unseen 2D recipe's LJ AE endpoint R² is
-0.0108 ± 0.3422 over five seeds (all 20/20 validation networks), versus Reid
0.9028 and low-T 0.9581. It does not establish adequate LJ response preservation.
No new LJ propagator training is part of this study.

Eight variants × seeds 3456456/123/456/786/2026 = 40 AE-only runs. Every run
uses the corresponding excluded-mixed-T recipe and retained-source split IDs:
20 training/20 validation networks each from Reid, low-T, LJ; frames 0–100;
batch32, lr1e-4, weight decay1e-5, physical reference context, normalized
in-memory coordinates. Mixed-T is loaded only after checkpoint selection,
with zero training/test entries and the same 20 validation networks.

| Variant | Dimension / hidden | Change / selection |
|---|---|---|
| longer | 2 / 96 | Original node-pooled loss and val-loss selection; cap14→40, patience3→8 |
| equal_source | 2 / 96 | Equal source/graph reconstruction objective; worst-source reconstruction selection |
| response_selection | 2 / 96 | As equal_source; worst-source frame-100 p-ratio R² selection |
| wider | 2 / 128 | As equal_source, increase hidden width only |
| dimension4 | 4 / 96 | As equal_source, increase latent dimension only |
| dimension4_response | 4 / 96 | As dimension4; response selection |
| lj_edges | 2 / 96 | As equal_source; fifth LJ edge-indicator channel and LJ graph-distance-three augmentation |
| lj_edges_response | 2 / 96 | As lj_edges; response selection |

All new runs use cap40/patience8; early stopping may differ. Response-selection
variants explicitly use retained-source validation response labels for checkpoint
selection; gradient targets remain displacement reconstruction. Other variants
select using reconstruction loss. Mixed-T is not a selection source. The
`equal_source` comparison changes objective weighting and checkpoint criterion
together, so it is a recipe comparison rather than an isolated attribution.

Save per-trajectory/source/frame position MSE, axial/transverse strain errors,
p-ratio R²/MAE/target variance, valid/total counts at 5/10/25/50/75/100. Rank
fully completed five-seed variants on worst retained-source endpoint mean R²,
requiring all 20/20 valid predictions; preserve all per-source/frame results.
An improved rank alone is not proof the AE is fixed. Evaluate whether LJ
response improves reliably without losing Reid/low-T, and inspect earlier
frames and field/strain errors before advancing to dynamics. Mixed-T results
remain post-selection diagnostics.

Exact recipes, frozen training source/scripts and code SHA-256 manifest:
`notebooks/results/lj_ae_repair/`. Job ledger `jobs.jsonl`; collector
`scripts/aggregate_lj_ae_repair.py`. Jobs use `mendels_q`, shared single-node
placement, eight CPUs, 32 GB, no host pin. Source exclusion, retained split IDs,
configuration expansion, compilation and PBS syntax checks passed. No model
implementation was modified; these use existing supported AE training options.
No repair success claimed yet; results pending.

## Partial AE reconstruction review — 28/40 completed

Collected existing completed runs while remaining jobs run; no dynamics launched.
Three recipes have all five seeds: dimension4_response, lj_edges, and
lj_edges_response. Endpoint validation response R² (mean ± seed SD):

| Recipe | Reid | Low-T | LJ |
|---|---:|---:|---:|
| dimension4_response | 0.870 ± 0.168 | 0.968 ± 0.011 | 0.210 ± 0.285 |
| lj_edges | 0.950 ± 0.016 | 0.949 ± 0.025 | -0.165 ± 0.229 |
| lj_edges_response | 0.933 ± 0.030 | 0.954 ± 0.016 | 0.113 ± 0.030 |

All source/seed endpoints have 20/20 valid predictions. Other recipes are
incomplete; their means cannot establish a winner. These are AE reconstructions,
not rollouts, and no recipe yet establishes adequate LJ response preservation.

New saved-prediction diagnostics (`scripts/diagnose_lj_ae_predictions.py`)
write per-seed/source/frame and aggregate CSVs plus input SHA-256 provenance
under `lj_ae_repair/prediction_diagnostics*`. No calibration is fitted. For
completed dimension4_response, mean per-seed LJ bias-squared/MSE is 0.030,
response correlation 0.553, predicted/true response SD 0.726, and mean absolute
axial/transverse strain error divided by mean absolute target strain is
0.090/0.124 at frame100. Thus constant bias alone does not explain failure.
For lj_edges_response the SD ratio is 0.423 (compressed response variation),
correlation 0.492, and relative strain errors 0.103/0.119.

LJ response R² at frames5/10/25/50/75/100 is
-17.255/-4.991/-0.450/0.117/0.231/0.210 for dimension4_response and
-3.452/-1.201/-0.149/-0.069/0.114/0.113 for lj_edges_response.
Early small-strain ratios are especially weak. These descriptive results motivate
a controlled strain-aware reconstruction objective (avoid directly dividing by
near-zero axial strain) and comparison across frames, rather than assuming
more latent dimensions or endpoint checkpoint selection resolves the problem.
This is a follow-up hypothesis, not a verified mechanism or launched experiment.
Mixed-T remains post-selection diagnostic only; no final-test data touched.
The automatic collector now refreshes these diagnostics when the matrix ends.

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

Deeper architecture audit: `docs/research/shared_dynamics_deeper_audit.md`. Saved history diagnostics suggest LJ training error is also high (~0.30 normalized MSE vs ~0.33 validation); no irreducible-noise claim. Identified nonzero-mean standardized-edge reversal inconsistency in frozen model; algebraic check `scripts/audit_ae_edge_reversal.py`, downstream impact pending. Terra notified to isolate correction from architecture changes. Proposed encoder-vs-decoder latent optimization, latent-use controls, and predictive temporal objectives without expert supervision.

## Automatic AE reconstruction collection

Collected 40/40 completed runs. Source-wise results, seed SDs, counts,
field/strain errors and a continuation packet are under `notebooks/results/lj_ae_repair/`.
Read `review_packet.md` and the underlying recipes before making a scientific claim
or advancing to LJ dynamics. Automatic collection is not a declaration of success.

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

Review notebook: notebooks/latent_space/11_compact_lj_reconstruction.ipynb,
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
