# Research status

The project is an ongoing shared dynamics and ML-only engineering study. The
primary scientific metric is post-fit p-ratio R-squared from reconstructed or
autonomously predicted positions. The shared noisy-LJ studies remain validation
development; their reserved final test is untouched.

The current priority is **noisy-LJ propagation through step 100**. Keep the
existing 2D message-passing AE frozen as the accepted reconstruction reference
and improve the shared predictor while retaining Reid and low-T performance.
Mixed-T remains excluded from training and checkpoint selection, with transfer
evaluated afterward. See the [dynamics goal](research/dynamics_goal.md).

Shared AEs retain useful response information for Reid and dePablo sources.
The completed five-seed checkpoint backfill found standard shared 2D LJ
endpoint R-squared of 0.451 ± 0.020 at frame 100 and 0.413 ± 0.269 at frame
199; message-passing 4D reached 0.633 ± 0.095 and 0.127 ± 0.284. Spatial
decoding has not consistently improved this response metric. Each estimate uses
five seeds on the same 20 validation networks: 100/100 valid evaluations per
horizon, with seed mean ± sample SD. These are AE reconstruction scores, not
autonomous rollout scores. The full,
source-wise evaluation and estimator conventions are in the
[p-ratio review](../notebooks/results/compact_lj_pratio/review.md).

The current matched three-seed MP2D reconstruction reference at step 100 is
0.594 ± 0.064 for noisy LJ, with 20/20 valid networks per seed. This is accepted
for the current propagator-focused phase; autonomous LJ response remains below
the target. Historical late-horizon reconstruction limitations are retained as
evidence, but extending to step 199 is not the immediate requirement. The
[results index](../notebooks/latent_space/experiment_results_index.md) records
completed matrices, failures, and the next evidence required.

Joint-Adam engineering gives development evidence, not held-out reliability.
The frozen-model procedure, final-design boundary, and required reporting are
defined in [engineering_goal.md](research/engineering_goal.md); detailed runs
and controls are in the [engineering log](../notebooks/latent_space/network_engineering_log.md).

Historical execution plans, architecture audits, and positive controls are
preserved in [past_experiments/research](past_experiments/research/).
