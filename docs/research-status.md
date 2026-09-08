# Research status

The project is an ongoing shared ensemble/CV study, with source-wise evidence and frozen result artifacts retained under `notebooks/results/`. No final-test conclusion is available.

The completed reference-simplification matrix fit Reid and dePablo low-T and evaluated mixed-T only post-fit. With a 4D latent and 16D graph context, reducing the per-node reference from 96D to 16D changed autonomous frame-199 coordinate MSE by +2.55% (Reid), -0.14% (low-T), and -0.04% (mixed-T); 8D was similarly close. At 2D, a 16D reference cost 15.24% on low-T. Removing graph context worsened every paired source/seed comparison. Evidence: `notebooks/results/reference_simplification/collection_code_v2/`; recipe and caveats: `notebooks/latent_space/reference_simplification_log.md`.

Noisy-LJ remains the open reconstruction problem. The five-seed near-08 bridge had frame-149 LJ coordinate MSE around 2.09–2.13e-5 across latent dimensions 2/4/6/8, so increasing latent width alone did not resolve it. The bridge has a different width, token budget, prefix coverage, and edge recipe, so it is not a matched compact-reference comparison. Evidence: `notebooks/results/lj_ae_08_bridge/`, `docs/research/compact_reference_lj_next_plan.md`.

The compact-LJ code-v2 matrix uses full 200-frame fitting, 16D reference priority, shared Reid/low-T/LJ and LJ-only controls, and mandatory LJ distance-3 edges. Its AE results and provenance are in `notebooks/results/compact_lj_reconstruction/`; interpret shared versus LJ-only cautiously because normalization and exposure differ. A compact motion audit found comparable sampled displacement scale and high increment persistence in Reid, low-T, and LJ; per-network PCA is only an optimistic descriptive diagnostic, not AE performance. Evidence: `notebooks/results/compact_lj_motion_audit/`.

Frozen-code fitted-latent and spatial-residual checks show modest same-target improvement from optimizing codes and that late LJ coordinate residuals are mainly non-affine under a post-fit affine decomposition. They do not establish decoder global optima, generalization, noise irreducibility, or a causal explanation. Evidence: `notebooks/results/compact_lj_fitted_codes/` and `notebooks/results/compact_lj_spatial_residuals/`.

Next work should retain source-wise valid/total counts, exact manifests and hashes, match 2D/4D and 6D/8D capacity tests where relevant, and establish graph-specific full-horizon reconstruction before selecting a latent propagator for LJ. Do not use expert response quantities in fitting or checkpoint selection.
