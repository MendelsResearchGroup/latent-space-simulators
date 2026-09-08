# Research conventions

Learn from observed trajectories and network structure. Do not add p-ratio, strain, expert-derived slow observables, or physics-informed targets to inputs, objectives, or model selection. Evaluate response quantities after fitting. Keep historical response-selected experiments clearly identified.

Before designing a sweep, consult the experiment results index and the relevant logs. Use hypothesis-driven comparisons and multiple seeds. Test latent dimension rather than assuming it: compare 2D/4D and, for noisy-LJ, 6D/8D when relevant. Preserve the required LJ graph-distance-three edge augmentation across training and inference.

Record exact configuration, data splits, seeds, source hashes, job IDs, status, and source-wise metrics with valid/total counts. Keep failed and negative results. Distinguish validation from final test; pooled metrics alone do not establish transfer.

PBS jobs use `mendels_q`, shared placement (`place=free:shared`), and no host pins. Ordinary attention-AE runs start at 8 CPUs and 16 GB; measure resource use for different workloads. Use the launchers in `hpc/` and keep each run's frozen code and recipe.

Keep notebook text short and plots inline. Use the Editorial palette from `lss.plotting`. Do not export figure files by default. Update the relevant experiment log when collecting a sweep, particularly `07b_experiment_log.md` for notebook 07b.
