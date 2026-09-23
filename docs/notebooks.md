# Notebook map

Current notebooks are grouped by purpose. Historical IDs remain in result
directories and logs to preserve provenance.

| Notebook | Purpose |
| --- | --- |
| [Dynamics 01](../notebooks/dynamics/01_mixed_dataset_shared_latent_space.ipynb) | Shared representation and post-fit mechanical probes. |
| [Dynamics 02](../notebooks/dynamics/02_mixed_dataset_shared_latent_rollout.ipynb) | Separate AE/propagator fitting and rollout evaluation. |
| [Dynamics 03](../notebooks/dynamics/03_four_source_standard_ae_pca.ipynb) | Standard AE/PCA analysis with structural LJ edges. |
| [200-MD-step dynamics](../notebooks/dynamics/200_md_step/README.md) | Dense-cadence 500-frame dynamics notebooks 01–03 and the low-T one-step-displacement AE comparison. |
| [Shared AE response and latents](../notebooks/dynamics/04_shared_ae_response_and_latents.ipynb) | Three-source reconstruction R² curves, raw 2D latent trajectories, and post-fit physical correlations. |
| [Compact LJ diagnostics](../notebooks/dynamics/diagnostics/compact_lj_reconstruction.ipynb) | Saved reconstruction and p-ratio results; no training. |
| [Response and rollout diagnostics](../notebooks/dynamics/diagnostics/compact_lj_response_rollout.ipynb) | Source-wise p-ratio R-squared by horizon. |
| [Engineering cohort](../notebooks/engineering/01_adam_engineering.ipynb) | Frozen-model engineering results and animations. |
| [Engineer one network](../notebooks/engineering/02_engineer_one_network.ipynb) | One frozen-model design, saved before final verification. |

Use the [experiment results index](../notebooks/latent_space/experiment_results_index.md)
and associated logs for exact recipes, seeds, artifacts, and current study
status. The CV and other retired notebooks remain under
[`notebooks/past_experiments/`](../notebooks/past_experiments/).

The notebooks require the datasets listed in each notebook's setup cell.
Historical cache directories and unavailable legacy inputs are recorded in the
experiment logs; saved figures and outputs are evidence, not fresh executions.
