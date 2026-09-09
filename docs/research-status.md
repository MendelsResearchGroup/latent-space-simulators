# Research status

The project is an ongoing shared dynamics and ML-only engineering study. The
primary scientific metric is post-fit p-ratio R-squared from reconstructed or
autonomously predicted positions. All conclusions remain validation findings;
the reserved final test is untouched.

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

Noisy-LJ reconstruction and long-horizon response remain open. A five-seed
training/validation audit found late response generalization uncertainty, and
the matched 2D/4D comparisons do not yet establish a robust full-horizon LJ
solution. The [results index](../notebooks/latent_space/experiment_results_index.md)
records completed matrices, failures, and the next evidence required.

Joint-Adam engineering gives development evidence, not held-out reliability.
The frozen-model procedure, final-design boundary, and required reporting are
defined in [engineering_goal.md](research/engineering_goal.md); detailed runs
and controls are in the [engineering log](../notebooks/latent_space/network_engineering_log.md).

Historical execution plans, architecture audits, and positive controls are
preserved in [past_experiments/research](past_experiments/research/).
