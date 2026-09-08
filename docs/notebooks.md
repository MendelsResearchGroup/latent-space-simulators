# Notebook map

The main sequence starts at 01. Historical experiment IDs, cache paths, and log filenames retain their original numbers so recipes and saved results remain traceable.

| Main notebook | Historical number | Purpose |
| --- | --- | --- |
| [01 — Shared latent space](../notebooks/latent_space/01_mixed_dataset_shared_latent_space.ipynb) | 06 | Shared representation, latent trajectories, and post-fit mechanical probes. |
| [02 — Shared latent rollout](../notebooks/latent_space/02_mixed_dataset_shared_latent_rollout.ipynb) | 06b | Separate AE and propagator, rollout curves, and reconstruction ceiling. |
| [03 — Standard AE/PCA](../notebooks/latent_space/03_four_source_standard_ae_pca.ipynb) | 09 | Ordinary reconstruction-selected AE with LJ structural edges and post-fit analysis. |

Saved outputs are historical evidence; use the [results index](../notebooks/latent_space/experiment_results_index.md) for the latest multi-seed findings. Expert response quantities in these notebooks are post-fit analyses, not model-selection targets.

## Current diagnostics

[Compact LJ reconstruction](../notebooks/latent_space/diagnostics/compact_lj_reconstruction.ipynb), formerly 11, reads saved reconstruction runs, observed-motion baselines, and frozen-decoder code-fitting diagnostics. It does not train or use p-ratio. Its status table reflects local artifacts, not the live scheduler.

## Removed experiments

Legacy 04, 06c, both 06d notebooks, 07, 07b, 08, and 10a/10b were deleted from the working tree. Their notebooks remain recoverable from Git history; experiment logs, result artifacts, and reusable model implementations remain in place.

Historical qualifications still apply: noisy-LJ 06d and 08 used expert response for checkpoint selection; the matched-GNN 06d included response-based comparison inputs. The removed 07b configured seed `4654562` despite a seed456 checkpoint/model name and matched-split log entry. The removed 10a/10b claimed matched training budgets but configured 15 versus 5 epochs and had no saved outputs or corresponding result directories available. These discrepancies were documented, not repaired by cleanup.

Historical CV notebooks remain in the separate `notebooks/baselines/` directory: `01_train_cv_transformer.ipynb`, `01b_compare_rollout_cv_vs_transformer.ipynb`, and `01c_cv_count_simulator_sweep.ipynb`. They contain historical response fitting/readouts; 01b and 01c explicitly use response-based checkpoint selection. They are not current state-only model-selection recipes.

## Assets required by the notebooks

The current latent-space notebooks use these locally supplied files under `data/`:

- `reid_200_frames.pt`
- `depablo-near-zero-temp.pt`
- `depablo-10k-mix-temp.pt`
- `lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt`

`2340_dePablo_networks_OOL_undirected.pt` is also supplied. The CV baseline input `data_aux_opt_lowT_448sims_noang_bidirect.pt` is unavailable.

The exact historical notebook cache directories for the main sequence are unavailable. Preserved figures are historical outputs; reproducing them requires the missing checkpoints or fresh training. Cache matching checks remain enabled. Renumbering leaves code, outputs, and cache paths unchanged.

The compact-LJ result directories used by the diagnostics reader are included. Separate sweep checkpoints and tables are listed in [results.md](results.md).
