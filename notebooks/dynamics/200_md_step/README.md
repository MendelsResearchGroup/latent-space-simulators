# Dynamics with 200 MD steps between stored frames

These notebooks use 500 stored frames (0–499), separated by 200 MD steps and ending at MD step 99,800:

1. [One trajectory per source](01_one_trajectory_per_source.ipynb): the notebook-01 plot sequence for a strict low-T-only, one-network-per-family AE, using display seed 456 and retaining the three-seed summary; no rollout claim.
2. [Twenty trajectories and rollout](02_twenty_trajectories_and_rollout.ipynb): the notebook-02 plot sequence reproduced with 200-MD data—initial states, mixed-T and matched low-T paths, shared response, early latent–physical response, dimensionality, and rollout.
3. [Latent morph and attribution](03_latent_morph_and_attribution.ipynb): the notebook-03 plot sequence reproduced with 200-MD data—direct descriptors, initial response, static descriptors, latent response, edge importance, and matched early response.
4. [One-step displacement AE](04_one_step_displacement_ae.ipynb): low-T-only, one-versus-twenty-network AEs predicting the next stored-frame increment. Shows initial latent/final-response association, matched raw latent paths together on shared axes, motion slopes, and teacher-forced one-step diagnostics.

The notebooks directly under [`notebooks/dynamics/`](../) remain the original 1,000-MD-step study: 200 stored frames ending at MD step 199,000. The two studies therefore differ in both saved-frame cadence and total physical horizon.

Notebook 01 reads the dedicated frozen low-T-only artifacts in `notebooks/results/mini_temperature200_lowt_one`; notebooks 02–03 read `notebooks/results/mini_temperature200_frames500`; notebook 04 reads `notebooks/results/mini_temperature200_one_step_ae`. They do not retrain models or use p-ratio in model fitting or checkpoint selection.
