# Notebook map

| Notebook | Purpose | Historical expert-observable status |
| --- | --- | --- |
| `notebooks/latent_space/06_mixed_dataset_shared_latent_space.ipynb` | Three-source shared latent-space study. | p-ratio appears in saved diagnostic plots only. |
| `notebooks/latent_space/06b_mixed_dataset_shared_latent_rollout.ipynb` | Shared latent rollout following notebook 06. | p-ratio is a saved rollout diagnostic. |
| `notebooks/latent_space/06c_shared_multiphysics_history_rollout.ipynb` | Frozen-AE shared history rollout. | p-ratio is a saved diagnostic. |
| `notebooks/latent_space/06d_matched_shared_gnn_rollout_comparison.ipynb` | Historical latent-propagator/spatial-GNN comparison. | Includes response-based historical comparison inputs. |
| `notebooks/latent_space/06d_noisy_lj_history_rollout.ipynb` | Historical noisy-LJ history rollout. | **Incompatible historical recipe:** p-ratio was used for AE HPO/checkpoint selection. |
| `notebooks/latent_space/07_noisy_lj_normalized_context_rollout.ipynb` | Noisy-LJ normalized-context rollout. | p-ratio is a saved diagnostic. |
| `notebooks/latent_space/07b_mixed_reid_depablo_lj_context_ablation.ipynb` | Shared-source context ablation. | p-ratio is a saved diagnostic. |
| `notebooks/latent_space/08_four_source_nash_mtl_autoencoder.ipynb` | Historical three-source AE study with LJ structural edges. | **Incompatible historical recipe:** p-ratio checkpoint metric. |
| `notebooks/latent_space/09_four_source_standard_ae_pca.ipynb` | Standard AE/PCA reconstruction analysis. | p-ratio is a post-fit analysis observable. |
| `notebooks/latent_space/10a_individual_direct_autoencoder_simulators.ipynb` | Per-source direct AE simulators. | p-ratio is a saved diagnostic. |
| `notebooks/latent_space/10b_shared_direct_autoencoder_simulator.ipynb` | Shared direct AE simulator. | p-ratio is a saved diagnostic. |
| `notebooks/latent_space/11_compact_lj_reconstruction.ipynb` | Compact-reference noisy-LJ reconstruction results reader. | No p-ratio use in this notebook. |
| `notebooks/latent_space/archive/04_latent_space_simulator_legacy.ipynb` | Archived antecedent latent simulator. | Contains post-fit p-ratio readout analysis. |
| `notebooks/baselines/01_train_cv_transformer.ipynb` | Historical CV-transformer workflow. | Historical response readouts; do not use its p-ratio fitting as compliant model selection. |
| `notebooks/baselines/01b_compare_rollout_cv_vs_transformer.ipynb` | Historical rollout comparison. | **Incompatible historical recipe:** stated p-ratio checkpoint selection. |
| `notebooks/baselines/01c_cv_count_simulator_sweep.ipynb` | Historical CV-count sweep. | **Incompatible historical recipe:** validation rollout p-ratio checkpoint selection. |

## Assets required by the notebooks

Raw data expected under `data/`:

- `reid_200_frames.pt`
- `depablo-near-zero-temp.pt`
- `depablo-10k-mix-temp.pt`
- `lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt`
- `2340_dePablo_networks_OOL_undirected.pt`
- `data_aux_opt_lowT_448sims_noang_bidirect.pt`
- legacy archive inputs: `200_rand_pruned_OOL_bidirect_val1057.pt`

The first four current latent-space data assets plus `2340_dePablo_networks_OOL_undirected.pt` are included in the local project. `data_aux_opt_lowT_448sims_noang_bidirect.pt` and `200_rand_pruned_OOL_bidirect_val1057.pt` are not available, so the CV baselines and archived 04 notebook need those assets supplied separately to execute.

The exact historical notebook cache directories for 06–10 are unavailable. Their saved figures and code are preserved; reproducing those outputs requires the missing checkpoints or a fresh training run. Cache checks remain enabled, so moving paths can require retraining.

The compact-LJ result directories used by notebook 11 are included. Separate sweep checkpoints and tables are listed in [results.md](results.md).
