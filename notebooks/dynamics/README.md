# AE and propagator

Edit these notebooks directly. Shared plotting lives in `src/lss/plotting.py`;
old builder/patch scripts are archived as records under
`tools/past_experiments/notebook_writers/`, not maintained notebook sources.

The numbered notebooks in this directory use the original **1,000 MD steps between stored frames** data. The matched dense-cadence rerun is kept separately in [`200_md_step/`](200_md_step/README.md): 500 frames separated by 200 MD steps.

1. [One trajectory per source](01_mixed_dataset_shared_latent_space.ipynb): executed 1,000-MD-step four-source 2D latent analysis, emphasizing heldout initial-z/final-p ordering.
2. [Twenty trajectories per source](02_mixed_dataset_shared_latent_rollout.ipynb): executed one-versus-twenty geometry comparison, combined complete low-T z0/z1 trajectories, and source-wise AE/autonomous-rollout p-ratio R² versus MD horizon. Displayed R² curves stop at zero; tables preserve exact values.
3. [Exact one-versus-twenty comparison](03_four_source_standard_ae_pca.ipynb): direct z0/z1 redundancy, the completed multi-seed 1/2/4/8/10/12/14/20 latent morph, early-turn dynamics, node/stiffness family separation, static graph-input attribution, and matched per-edge sensitivity maps through both encoders. The filename is historical; the notebook no longer replaces the learned latent variables with PCs.

[Response and rollout diagnostics](diagnostics/compact_lj_response_rollout.ipynb) show post-fit p-ratio R² against horizon, with reconstruction and propagation separate. [Reconstruction diagnostics](diagnostics/compact_lj_reconstruction.ipynb) inspect the AE ceiling.

[A latent crossing](10_latent_crossing_and_response.ipynb): held-out low-T crossings in the 1,000-MD one- and twenty-network AEs; raw latent paths, initial position plus movement, and ground truth versus AE response. Includes a same-time near-meeting search and cohort reconstruction accuracy.

Saved figures describe historical runs; see the [results index](../latent_space/experiment_results_index.md) for artifacts, provenance and limitations. Missing historical caches require recovery or explicit retraining.

4. [Shared AE response and latents](04_shared_ae_response_and_latents.ipynb): shared message-passing 2D reconstruction R² for Reid, low-T dePablo and LJ, raw latent trajectories, initial/final response associations, and physical correlations.

5. [Mini networks: stiffness and node optimization](05_dense_reid_small_steps.ipynb): separate dataset sections, matched 20/20/80 splits and frames 0–99; ordinary/frame-balanced AE training with one-step propagators. Shows p-ratio R² from 0–1 and physical rollout MSE.

6. [Shared propagator through100 steps](06_shared_propagator_100_steps.ipynb): frozen2D AE, full autonomous rollout training with latent or decoded-position loss; cached source-wise validation p-ratio R² and run status. Run after the study collector completes.

7. [Broad propagator comparison](07_broad_propagator_100_steps.ipynb): six temporal models with mean or learned-attention reference context; frozen2D AE, physical response at100steps, all36runs. Run after collection.

8. [Observed initial history](08_observed_history_100_steps.ipynb): matched frame0 versusframes0–2 conditioning, forecast-only p-ratio R² through100, frozenAE andlinearlatentextrapolation references. Reads cachedcollection.

All active dynamics and engineering notebooks start with `PLOT_THEME = "light"`; change it to `"dark"` and rerun the plotting cells to switch the shared Editorial theme.

[Mini shared latent dynamics](06_mini_shared_latent_dynamics.ipynb): single-network fitting and mixed-temperature transfer, then stiffness-only versus joint-original 2D message-passing AEs with frozen one-step propagators. All200 saved frames, source-wise AE/rollout p-ratio R² and MSE, and matched held-out latent/response associations. Reads queued study caches; mixed-T is evaluation-only.

[Mini data: original and mixed temperature](07_mini_data_original_and_mixed_temperature.ipynb): ground-truth paired three-second GIFs, response-versus-frame curves, endpoint/full-trajectory-fit p-ratio histograms for all200networks per source, and paired response scatter. No models. GIFs are stored alongside the notebook in its title-named folder.

## Separate paused side project

[`cold_graphs/`](cold_graphs/README.md) contains Sergey's unrelated three-dimensional cold-rubber study. It does not use the 200- or 1,000-MD-step datasets, is not imported by the numbered workflow above, and should not be used to interpret notebooks 01–09.
