# Research conventions

The engineering objective is reliable gradient-guided editing of an original network through the frozen ML model, preferably the AE alone, with the learned propagator included if needed. LAMMPS is final verification only. See `docs/research/engineering_goal.md` for the optimization boundary and success criteria.

Learn from observed trajectories and network structure. Do not add p-ratio, strain, expert-derived slow observables, or physics-informed targets to inputs, objectives, or model selection. Evaluate response quantities after fitting. Keep historical response-selected experiments clearly identified.

Post-fit p-ratio R² is the primary metric of scientific success. Compute it from the model's reconstructed or predicted positions, with the same declared estimator for reference and prediction. Position MSE is a secondary diagnostic; a compact latent may tolerate higher coordinate error while preserving the response of interest. Coordinate error alone is insufficient to reject an AE or simulator. Keep AE reconstruction, initial-response prediction, and autonomous rollout R² as separate claims. Report source-wise R² at explicit horizons, per-seed results and mean/SD, valid/total counts, estimator/coordinate conventions, and validation versus final-test status. Do not silently omit undefined p-ratios or choose a favorable estimator after looking at scores.

Every reconstruction or rollout study must include this post-fit evaluation of the frozen, state-selected checkpoints. Backfill missing p-ratio results from existing artifacts before launching replacement training; recover missing checkpoints first and rerun only irrecoverable cells with their exact recipes. High coordinate error is not by itself a reason to rerun or redesign a model.

Before designing a sweep, consult the experiment results index and the relevant logs. Use hypothesis-driven comparisons and multiple seeds. Test latent dimension rather than assuming it: compare 2D/4D and, for noisy-LJ, 6D/8D when relevant. Preserve the required LJ graph-distance-three edge augmentation across training and inference.

Record exact configuration, data splits, seeds, source hashes, job IDs, status, and source-wise metrics with valid/total counts. Keep failed and negative results. Distinguish validation from final test; pooled metrics alone do not establish transfer.

PBS jobs use `mendels_q`, shared placement (`place=free:shared`), and no host pins. Ordinary attention-AE runs start at 8 CPUs and 16 GB; measure resource use for different workloads. Use the launchers in `hpc/` and keep each run's frozen code and recipe.

Keep notebook text short and plots inline. Use the Editorial palette from `lss.plotting`. Do not export figure files by default. Update the relevant experiment log when collecting a sweep, particularly `07b_experiment_log.md` for notebook 07b.

Engineering uses ML-only optimization, with LAMMPS reserved for final verification. Predeclare the frozen model, objective, constraints, optimization budget, and final selection rule. Start from original networks, never simulator-selected engineered starts. No simulator-dependent proposal scoring, acceptance, stopping, restarts, or winner selection is allowed. Save a manifest with final-design hashes before verification; report every declared design and retain failures. Final verification must not feed back into the same optimization or turn previously checked designs into a fresh blind evaluation. Identify historical simulator-guided results and response-calibrated latent proxies explicitly. Keep the positions/dynamics-only model-training rule; any engineering objective derived from predicted positions is downstream of the frozen model. Reconstruction accuracy alone does not establish reliable engineering under interventions.

Use `lss.dynamics` for AE/propagator work and `lss.engineering` for network optimization, final evaluation, and animation. Retired implementations belong in `lss.past_experiments`. Current code uses canonical imports without runtime aliases. Convert historical checkpoint working copies explicitly with `tools/migrations/checkpoint_imports.py`; preserve the original files and conversion records. See `docs/project_structure.md`.

Write new workflow APIs with concrete required inputs. Let missing files, keys, and invalid shapes fail directly rather than adding `None` fallbacks, placeholder results, or catch-all recovery. Keep actual optimization constraints explicit. Do not rewrite frozen historical snapshots during cleanup.
