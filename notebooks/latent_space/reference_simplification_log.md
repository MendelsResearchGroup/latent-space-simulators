# Reference and propagator-context simplification

## Objective and prior evidence — 2026-09-07

Test the user-authorized model simplifications independently and together:
remove graph context from the propagator; reduce the retained per-node
reference representation from96 to16 or8; combine both. Keep the dynamic
hidden width96 and decoder token count32 fixed. Compare latent2D and4D and
five seeds3456456,123,456,786,2026. Three reference widths × two latent widths
× five seeds =30 AE recipes, each followed by two frozen-AE propagators
(context16 and none), giving60 dynamics fits. These are paired context tests,
not60 independent AE seeds.

Prior evidence consulted: experiment_results_index.md and07b_experiment_log.md
AD no-context screen, AI context4 screen, AP/AQ fixed-history controls.
AD's no-context failure used a different3-source AE and a fixed observed window;
AI reduced graph-level context, not per-node reference features. Neither settles
this two-source matched comparison. Existing single-stage architecture sweeps
also do not change per-node reference width. Preserve those negative results.
Historical response-selected propagator scores are not compliant selection
baselines under the current dynamics-only requirement.

## Implementation and controls

Baseline uses unchanged `attention`. `attention_reference8` and
`attention_reference16` retain R-dimensional features from ref_node_in; the
current-state encoder temporarily lifts these to96 before its existing sum
with evolving edge features. Decoder queries take R inputs; decoder MLP input
is96+R. Graph pooling receives only the retained R features and projects their
mean to16 when context is enabled. With context removed, propagator input is
only the current latent. There is no hidden96D reference bypass into the
reduced decoder or graph-context path. The static feature is affine in local
reference position and mean edge attributes, so reducing its width need not
remove96 independent physical degrees of freedom; this also tests a simpler
parameterization rather than presuming information loss.

The encoder attention pyramid, latent token decoder, legacy orientation
convention, dynamic width96, frame sampling, optimizers and data identities are
held fixed. The computation-preserving decoder K/V optimization discussed in
the meeting guide is not bundled into this architectural ablation. Existing
engineering jobs retain their frozen code and artifacts.

New reference variants are registered in the common model factory; training
and saved-bundle loading obtain the propagator's raw context width from the
actual reference representation. Ten targeted reference/gradient/save-load,
pipeline and engineering-export tests passed. Integration smoke jobs cover
R96/D2, R8/D2 and R16/D4, both contexts and full cache reload; their recipes and
any failures remain saved under the code version.

## Exact experiment recipe

- Authoritative manifest: notebooks/results/lj_ae_08_bridge/split_manifest.json.
- Fit sources: Reid and dePablo low-T only. First20 train IDs/source from that
  manifest; all20 validation IDs/source; final test lists empty.
- Mixed-T: fixed20 validation IDs loaded only after both context checkpoints
  have been selected. Report as post-fit transfer validation, not final test.
- Scope excludes noisy-LJ fitting; this experiment does not settle LJ capacity.
- AE: all200 training/validation frames, max40 epochs/patience8,
  AdamW lr1e-4/weight decay1e-5, source_mean gradients, worst-source validation
  coordinate reconstruction selection. No response callback/selection.
- Propagator: delta_mlp with64-unit hidden layers, one-step latent-delta loss,
  all199 adjacent transitions/network, max40 epochs/patience8,
  lr1e-4/weight decay1e-5, equal-source loss. No temperature/source IDs,
  progress, expert targets or physics loss. Explicit checkpoint_metric=None
  chooses validation latent loss; response callbacks disabled.
- Both contexts use an identical AE weight hash; AE weights are frozen and
  checked after dynamics fitting. Same propagator seed=AEseed+10000; different
  input widths mean this is matched-seed training, not identical initialization.
- Post-fit metrics: first-to-last response and final coordinate MSE only,
  source-wise valid/total counts, AE reconstruction and autonomous rollout.
  Real runs predict from frame0 to199 with no supplied future observations.
- Existing AE reuse requires an exact full configuration match. The engineering
  AE used only the first101 frames and is not eligible for this corrected
  full200-frame matrix. All30 recipes therefore fit a new AE.
- PBS: mendels_q, one shared node,8CPUs/16GB, no host pins.30 jobs each perform
  their two context comparisons sequentially. Source-wise automatic collector
  preserves incomplete/failed runs and writes paired metric differences.

## Artifacts and status

All jobs, recipes, split identities, source hashes, frozen source and status:
`notebooks/results/reference_simplification/`. Primary scripts:
`run_reference_simplification.py`, `submit_reference_simplification.py`,
`aggregate_reference_simplification.py` under scripts/.
Source-wise outcome tables and pairings are collected into collection_CODEVERSION.
A failed smoke is not a scientific outcome, and an incomplete sweep does not
establish that a simplification works. Success must be assessed across both
fit sources, post-fit mixed-T transfer, coordinates and response diagnostics.


### Full-span correction before the scientific sweep

Final review found `filtered_frame_ids(max_frames=101)` selects a prefix rather
than uniformly sampling the full200-frame trajectory. Code_v1 integration
smokes passed both context settings, model reload and post-fit transfer, but
full-run submission was stopped and its jobs cancelled before interpreting
results. Preserve those attempts in cancelled_prefix_runs.json and
attempt_status.jsonl. Code_v2 uses all200 AE training/validation frames, matching
the frame0→199 evaluation horizon and199 propagator transitions. No shortcut
reuse of the prefix-trained engineering AE. Re-run integration smokes on the
corrected snapshot, then launch the matched30-recipe/60-propagator matrix.

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

Automatic collection code_v2: 30/30 completed; 0 failed, 0 incomplete. Source-wise counts, seed aggregates and paired differences: `notebooks/results/reference_simplification/collection_code_v2`. No pooled-only selection.

### Completed reduced-reference comparison — 2026-09-07

All30 full-span code_v2 AEs and60 paired propagators completed (five seeds,
reference96/16/8 × latent2/4 × propagator context16/0). Collector4671446 failed
with duplicate status keyword; repaired live collector and recollected without
changing frozen training code or its failed collector log.270 source-wise rows;
each configuration/source has100/100 valid and coordinate-valid evaluations
(20 identical validation networks × five seeds, not100 unique networks).
Exact recipes, split identities, job IDs, frozen hashes and seed metrics remain
under reference_simplification/. Results: collection_code_v2/source_aggregate.csv,
paired_differences.csv, source_seed_results.csv and coordinate_comparisons.csv.

Coordinate conclusions, using autonomous frame0→199 final position MSE:
- Removing the propagator16D graph summary worsens all90 paired source/seed
  comparisons. Source/config mean increases range36.5–101.8%. This is useful
  structure information for this delta-MLP recipe, not evidence that a universal
  latent-only propagator is impossible.
- With4D latent and graph context16, reducing per-node reference96→16 changes
  mean MSE by +2.55% Reid, -0.14% low-T, -0.04% mixed-T; reference8 changes it
  by +0.48%, +0.87%, +1.88%, respectively. Promising simplification; five seeds
  do not establish formal equivalence. AE reconstruction/transfer remains
  seed-sensitive, and endpoint rollout alone does not settle entire trajectories.
- With2D latent, reference96→16 costs +0.90% Reid, +15.24% low-T, +5.70% mixed-T;
  reference96→8 costs +2.39%, +16.27%, +6.45%. Keep96/2 as smaller-latent control;
  do not assume4D is universally necessary.
- Reducing per-node stored reference96→8 is12× narrower, but the2D AE parameter
  count only falls172036→147292 (~14.4%); hidden width96 remains. Reference16 has
  150388 parameters. No measured training-speed benefit established here.

Post-fit p-ratio diagnostics agree that removing graph context hurts: Reid mean
R² becomes negative for every no-context configuration. These are diagnostics,
not training objectives or checkpoint/model-selection criteria.
Fit sources were Reid+low-T only; mixed-T is post-fit transfer validation, seen
by neither AE nor propagator during fitting. No noisy-LJ or final test used.
Recommended direction from coordinate evidence: keep graph context, carry compact
reference8/latent4 and baseline reference96/latent2 into full-trajectory diagnostics
before changing the default. Need test reconstruction throughout time and source-wise
LJ capacity separately; this matrix does not settle either.
