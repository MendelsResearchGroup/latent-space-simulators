# Compact-reference shared LJ research: evidence and next comparisons

2026-09-07. Proposed next experiments; not submitted by this review.
User accepts a modest low-T tradeoff and prefers16D per-node reference.
Retain16D propagator graph context. Do not confuse static node-reference width
with evolving latent dimension or total AE parameter count.

## Evidence used

- reference_simplification/code_v2:30 AEs/60 propagators, five seeds. R16/D2
  endpoint coordinate error vs R96/D2: Reid+0.90%, low-T+15.24%, mixed-T+5.70%.
  R16/D4 vs R96/D4: +2.55%/-0.14%/-0.04%. No LJ fitted in this comparison.
- lj_ae_08_bridge:40/40 complete,30/30/60 training networks,20 validation/source,
  five seeds. Near08 D6,width192,tokens46,150-frame prefix,50epochs. Frame100 LJ
  coordinate MSE7.9220e-6; frame149 (last trained frame)2.1299e-5. Matched D2/4/8
  frame149 MSE2.0941e-5/2.1077e-5/2.0842e-5: latent enlargement alone produces
  little change. Source_mean2.0573e-5. No shared recipe established reliable LJ
  reconstruction across time. Absolute MSE alone cannot quantify residual noise.
  Post-fit near08 LJ response R²=.329±.221 at100 and .036±.361 at149; strictly
  diagnostics, not selection. This was AE-only, not a trained shared LJ propagator.
- Bridge is not directly matched to compact sweep: larger training budgets,
  width, tokens, prefix coverage and edge schema. It adds graph-distance≤3
  synthetic relations only for LJ and an edge-relation channel. That channel
  identifies synthetic relations (not p-ratio); source-conditioned construction
  explains part of the recipe mismatch. User explicitly requires this augmentation
  for LJ in the next study: keep missing distance2/3 pairs, raw stiffness0 and
  relation indicator1, with geometric vectors/lengths. Existing springs retain
  stiffness and indicator0; other sources use the same five-channel schema with
  indicator0. Apply consistently at training/validation/inference. Do not omit
  LJ edges or interpret the older four-channel study as an equivalent control.
- shared_ae_architecture/code_v5:30/40 complete. MP2/D2 has five completed seeds,
  frame100 coordinate MSE low-T2.4561e-6±1.1669e-7, Reid6.5919e-6±1.3170e-7,
  LJ8.2516e-6±1.3271e-7,100/100 valid per source (20 unique graphs ×five seeds).
  MP2 improves over its corrected attention control but does not establish an LJ
  solution. Ten missing cells terminate with directory-collision FileExistsError;
  not negative scientific outcomes. Earlier stages/log status were stale.
- Fitted-latent diagnostic previously showed only modest best-start improvement;
  no strong evidence that encoder alone is responsible. Noise floor is unknown.

## Next questions, in dependency order

1. Verify reconstruction of graph-specific dynamics before new rollout training.
   Evaluate coordinate residuals across the full declared time range, on train
   and validation separately, plus mean-displacement/wrong-time/shuffled-code
   controls. A low coordinate average is insufficient if graph-specific motion
   is lost. Tiny training-subset fit separates optimization/decoder capacity from
   transfer. No expert observables in these decisions.
2. Compact shared AE:16D node reference, latent2/4/6/8, retaining identical
   attention backbone and a matched wider-reference control. Fit full observed
   horizon, not a101/150-frame prefix evaluated as full-horizon training.
   Require LJ distance3 augmentation and edge_feature_dim=5 in every matched
   LJ cell; preserve original spring stiffness and fixed reference connectivity.
   Use authoritative bridge split:30 Reid/30 low-T/60 LJ,20 validation each;
   mixed-T remains post-fit transfer, revised120-LJ final reserve untouched.
   Compare LJ-only and shared fitting with identical LJ IDs and matched LJ
   optimizer exposure, seed, normalization and evaluation coordinates. Shared
   normalization fitted on the declared training pool may be frozen for this
   diagnostic, explicitly recorded; LJ-only means LJ-only weight optimization.
   A no-LJ shared control isolates harm to Reid/low-T under the same protocol.
   Five paired seeds3456456/123/456/786/2026. Never rank by response diagnostics.
3. If joint fitting hurts relative to matched LJ-only, test separate AE warm-up
   followed by shared coordinate-based fine-tuning, with update-matched scratch
   controls. If both struggle, prioritize a richer node decoder or the already
   motivated nonlinear message encoder; do not blindly repeat width sweeps.
   Any corrected edge-reversal behavior must be identical in the new matched
   cells, with old frozen results retained as historical controls.
4. Once AE state reconstruction is adequate, freeze it and fit graph-conditioned
   latent dynamics. Compare the existing one-step delta-MLP with training on short
   autonomous rollouts against observed future coordinates. A causal history
   branch is warranted only if one-step prediction suggests missing state; label
   its extra observation budget, and do not imply frame0-only equivalence.

Before submission, freeze exact matrix, exposure budget, coordinate-based
acceptance comparisons, code/data/split hashes and seeds. Save all outcomes,
counts and failures. Standard comparable jobs8CPU16GB shared mendels_q, no pins;
measure larger diagnostic workloads separately. Allow the user's accepted
low-T tradeoff explicitly; do not demand every source strictly improve.

## Authorized first execution matrix

The user explicitly requested execution and an additional data-motion audit.
First matrix:50 AE-only runs, ten variants × five seeds. Exact variants:
- Shared corrected attention, reference16, latent2/4/6/8 (20 runs).
- Shared corrected attention, reference96, latent4/8 (10 runs).
- Shared two-layer message passing, reference16, latent4/8 (10 runs).
- LJ-only corrected attention, reference16, latent4/8 (10 runs).

All use hidden96, decoder tokens32, LJ distance3 edges and common5-channel schema,
normalized_delta inputs/targets, physical reference context, audited position
normalization, batch32, all200 frames, AdamW1e-4/weight decay1e-5, max60 epochs,
patience12, source_mean loss and worst retained-source reconstruction selection.
No propagator or expert observable supervision/selection. Shared30/30/60 train
and20/source validation; LJ-only same60/20LJ IDs. Shared runs evaluate20mixed-T
post-fit; LJ-only does not load/evaluate mixed-T. Final reserve unchanged.

LJ-only statistics are fitted on LJ training data, not borrowed shared statistics.
It is an independent reconstructibility control, not a fully isolated causal
interference test. Save normalization scales and actual training exposure; any
advantage requires controlled follow-up before attributing it to gradient conflict.
A no-LJ matched-budget control and normalization-matched LJ-only experiment are
not in these first50; existing compact results remain historical controls only.

Post-fit coordinate diagnostics: frames0/5/10/25/50/75/100/125/149/175/199, first5
train networks/source, all20validation/source, mixed-T20 for shared only. Compare
AE reconstruction against zero displacement and cyclic same-source/same-time
latent swap; report IDs, errors, valid/total and baseline-relative scores.
Reference16 and corrected message-passing model factory extensions passed18
model/edge tests plus3 pipeline tests. Five unrelated legacy normalization tests
also fail on unchanged frozen code_v2; preserve their identities in
compact_lj_reconstruction/implementation_validation.json. Three PBS smokes gate
full launch; each checks LJ raw zero stiffness/original spring preservation,
idempotent augmentation, fitting and saved-model equivalence.

Parallel motion audit4672395 uses10fixed training networks/source, full200frames,
trajectory-relative coordinates, temporal correlations, non-affine residual
statistics and per-network in-sample PCA. These are descriptive diagnostics only.
No training objective incorporates the fitted affine residual or PCA codes.

## Updated spatial hypothesis from motion audit

The sampled LJ motion is smooth with low per-network temporal rank but larger
non-affine displacement than the other sources. The difficult part may be learning
the graph-dependent spatial pattern. Low rank with a separate fitted basis per
network does not imply one universal2D/4D encoder-decoder can recover every graph.
If LJ-only and shared models both retain affine-scale residuals, investigate
richer structural-reference encoding or graph-conditioned spatial decoding, rather
than only enlarging z or making the encoder deeper. Current50runs and matched
latent-swap/residual diagnostics must establish this before another architecture
branch. No affine/PCA supervision is proposed.

## Execution status

All50 first-matrix code_v2 jobs are submitted after three successful smokes.
Collector4672487 and coordinate-comparison job4672488 are dependency queued.
Data audits4672395/4672421 are complete. Full AE outcomes are pending; do not
claim LJ-alone reconstructibility or shared interference until reviewed.
