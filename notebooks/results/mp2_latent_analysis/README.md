# Shared MP2D latent analysis

[Open the notebook](../../dynamics/04_shared_ae_response_and_latents.ipynb).

The current presentation excludes mixed-T from every plot and displayed table,
showing 60 networks / 12,000 latent frames across Reid, low-T dePablo and LJ.
The complete underlying latent export below is preserved. Frozen-checkpoint reconstruction inference was extended to steps 1/5/10 as recorded in `early_response/`.
`presentation_code/` records the updated three-source plotting helper.

Frozen message-passing/reference16 AE, seed 786, two raw dynamic coordinates.
The representative seed was fixed before extraction; axes are not rotated,
standardized, projected with PCA, or combined across models.

The latent export contains all 200 frames of the same 20 validation networks
per source: Reid, dePablo low-T, noisy LJ, and post-fit mixed-temperature
transfer. This is 80 networks and 16,000 latent pairs. Final-test networks are
not evaluated. No training, response fitting, checkpoint selection, or physics
simulation was performed. The five-seed reconstruction curve combines existing steps 25/50/100/150/199 in
`compact_lj_response_rollout/response_seed_results.csv` with steps 5/10 from
`early_response/response_seed_results.csv`.

- `frame_rows.csv`: raw z0/z1, framewise physical p-ratio, and axial/transverse strain.
- `network_rows.csv`: initial latent, final p-ratio, and original spring-network descriptors.
- `initial_response_correlations.csv`: descriptive source-wise correlations, one model seed, 20 networks per coefficient.
- `metadata.json`: exact validation IDs, checkpoint/data/recipe/code hashes, estimator, counts and parity checks.
- `code/`: source snapshot used for extraction and plotting.
- `presentation.json`: executed-notebook and plotting-helper hashes.

Physical p-ratio uses the established endpoint directional-side estimator with
frame-zero physical reference groups (quantile 0.10), including for mixed-T to
match the existing R² evaluation. It is not a new thermal OLS estimator. All
400 values at the five recorded evaluation horizons exactly match the previous
physical targets. Each source has 3,980/4,000 defined frame p-ratios; its 20
frame-zero values are undefined and retained. Strains use changes in directional
side extents divided by their frame-zero extents.

Mean stiffness and degree describe original spring edges, excluding added LJ
relations. Framewise correlations compare networks at a fixed step, avoiding
pooled time-trend correlations. Initial-latent correlations target final physical
p-ratio at step 199. These are Pearson associations, not prediction R² or
held-out engineering evidence. Constant observables have undefined correlation.
Response-colored paths show steps 1–199 and clip only the color scale at the
pooled 2nd/98th percentiles; all trajectories remain included.

The export completed locally on one CPU before a proposed queue handoff; no
PBS job was submitted. The notebook now reads these tables and renders quickly
without repeating inference. No figure files are exported; plots are embedded
in the notebook.

The notebook also includes all-frame z0/z1 scatter plots against physical p-ratio and transverse strain, separated by source and colored by trajectory step. Per source, p-ratio has 3,980/4,000 defined frame pairs; transverse strain has 4,000/4,000. No temporal pooling coefficient is reported for these repeated observations.

Response-colored trajectories and within-step latent/p-ratio correlations now cover steps 1–199 (all 199 steps for correlations). The all-frame scatter retains every defined observation. The current five-seed reconstruction R² view begins at step 5; step1 evaluations remain in the saved results. Notebook re-executed with nine inline figures.

Early reconstruction evaluation used the same frozen five checkpoints, exact20 validation networks/source, physical-coordinate endpoint estimator and LJ augmentation. Step25 truth/prediction parity passed on all300 network/seed pairs (maximum absolute difference9.89e-17). All60 source/seed/horizon summaries have20/20 valid values; the45 new summaries cover steps1/5/10. Job4676381 completed with54s walltime and about2.45GB measured peak memory. Exact inputs, source hashes, code snapshots and execution provenance are under `early_response/`. No fitting, selection, physics or final-test evaluation. The plotted y-axis is linear between−1 and1 and logarithmic outside that interval; all seed mean±SD ranges remain visible.


Current view: the reconstruction R² plot starts at step5. The step1 results remain preserved in `early_response/` but are excluded from that plot. Updated plotting cells are intentionally unexecuted; run the notebook to refresh them. Rendering is separate from model evaluation.

A single compact figure now replaces the broad initial-response and network-property scans, displaying the initial LJ line versus vertical centroid and one latent-rate/final-p-ratio relationship per source. The full search is retained in [correlation_search/README.md](correlation_search/README.md), including all tested candidates, permutation correction, bootstrap intervals and outlier checks. No images were inspected and no notebook was executed for this update.
