# Compact LJ decoder follow-up

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
