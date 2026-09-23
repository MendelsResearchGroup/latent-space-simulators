# Noisy-LJ latent-influence edge pilot

The downstream model is identical across arms: a four-block acceleration GNN
with training-only feature/target standardization and LayerNorm in both
encoders and every residual node/edge block. The base graph already contains
the required noisy-LJ graph-distance-two/three augmentation.

| edges | post-fit p-ratio R² at 50 | post-fit p-ratio R² at 100 | valid |
|---|---:|---:|---:|
| base | 0.652 ± 0.053 | 0.397 ± 0.196 | 60/60 |
| size-matched random | 0.425 ± 0.296 | 0.096 ± 0.354 | 60/60 |
| latent influence | **0.696 ± 0.012** | **0.497 ± 0.027** | 60/60 |

Values are mean ± sample SD over downstream seeds 123, 456, and 786 on the
same 20 validation networks. Checkpoints were selected only by one-step
acceleration MSE. P-ratio is computed afterward from physical positions using
separate time-OLS fits with intercept for transverse and axial strain over
frames 0 through the stated horizon.

This is positive pilot evidence, not a final claim: edge discovery used one
fixed seed-123 global 2D latent simulator. Because that simulator is global,
the added graph comes from common physical-node participation in centered
multi-step latent perturbation responses, not a literal per-node latent
Jacobian.

Artifacts: `postfit_seed_summary.csv`, `postfit_summary.csv`,
`postfit_per_network.csv`, and `validation_added_edges.csv`.
