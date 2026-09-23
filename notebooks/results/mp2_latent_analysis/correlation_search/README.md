# Initial latent position and trajectory rates

Exploratory, post-fit analysis of frozen MP2D seed 786: the same 20 validation networks each from Reid, low-T dePablo, and noisy LJ. Mixed-T and final-test networks are excluded. No AE training, checkpoint selection, new model inference, or physical simulation was performed.

Initial-line coordinate is the projection of the source's initial raw z0/z1 onto its leading principal axis, oriented with positive z0 component. For LJ it accounts for 94.38% of initial latent variation. This analysis projection does not change the notebook's raw latent plots.

A trajectory rate is the ordinary least-squares slope of z0 or z1 versus step, with an intercept, over frames 0–199. A geometric path slope is separately defined as OLS z1 versus z0. Each correlation has 20 independent network pairs, not 4,000 independent frames.

The compact view retains four relationships:

| Latent quantity | Observable | Pearson r | Bootstrap 95% interval | Leave-one-network-out r range |
|---|---|---:|---:|---:|
| Initial LJ line coordinate | Mean initial physical y-position | -0.791 | [-0.934, -0.433] | [-0.868, -0.713] |
| Reid dz0/dstep | Final physical p-ratio | -0.969 | [-0.986, -0.941] | [-0.975, -0.961] |
| Low-T dePablo dz0/dstep | Final physical p-ratio | -0.994 | [-0.998, -0.988] | [-0.996, -0.993] |
| LJ dz1/dstep | Final physical p-ratio | -0.826 | [-0.934, -0.641] | [-0.864, -0.789] |

All have 20/20 valid pairs. The initial LJ relation suggests sensitivity to vertical placement or mass distribution; it does not establish a causal mechanism, translation artifact, or mechanical response prediction. Absolute latent signs and orientation belong to this model seed. Geometric z1-versus-z0 path slopes did not retain a significant association after search correction; their final-p-ratio correlations are approximately -0.31, +0.31, and -0.62 for Reid, low-T, and LJ respectively.

`all_correlations.csv` preserves all 612 candidate rows, including 12 constant/incomplete-observable rows. Multiplicity correction uses 20,000 within-source network permutations and the maximum absolute Pearson correlation over all 600 evaluated relationships. Selected adjusted p-values are 0.01155, 0.00005, 0.00005, and 0.00320 in the table order. Bootstrap intervals use 5,000 paired-network resamples; they are descriptive intervals after an exploratory search, not independent confirmation. Seed: 20260909.

`network_analysis.csv` contains the 60 plotting rows and complete descriptors. `selected_relationships.csv` supplies the four displayed coefficients; `candidate_relationships.csv` records the initial ranking. `metadata.json` records definitions, line directions, hashes, candidates and selection. Static descriptor definitions and exact input IDs/hashes are in `../structure_descriptors/metadata.json`. Source and analysis scripts are frozen in `code/`.

The notebook is a view of these saved tables. Its updated cells were syntax-checked, but not executed or rendered, as requested.

Framewise rate view: `frame_rate_correlations.csv` stores1170 within-source/within-step Pearson coefficients for Δz0 andΔz1 against ending-frame physical p-ratio at steps5–199. Backward differences are computed before filtering, so step5 uses z5−z4. All coefficients use20/20valid network pairs. Median r for z0/z1: Reid−0.921508/−0.756935; low-T−0.977302/−0.792287; LJ−0.561741/−0.638375. No smoothing or pooled-frame significance; frame p-ratio retains the established reference-frame-zero endpoint estimator. Notebook shows one2×3scatter figure colored by ending step, with median within-step r. This is distinct from one whole-trajectory fitted rate versus final response. Exact definitions and hashes in frame_rate_metadata.json; plotting cell remains unexecuted.
