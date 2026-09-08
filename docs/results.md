# Experiment results

`notebooks/results/` contains the available scientific experiment artifacts: split manifests, exact recipes, job records, source-wise tables, diagnostics, frozen source snapshots, and trained weights. The working copy includes large artifacts; Git ignores datasets, model weights, and historical frozen code directories. Small result tables and provenance remain available for version control.

| Study | Main evidence |
| --- | --- |
| `reference_simplification/collection_code_v2` | Matched node-reference 96/16/8, latent 2/4, and propagator-context 16/0 comparisons |
| `compact_lj_reconstruction/collection_code_v2` | Full-horizon shared and LJ-only AE reconstruction; capacity and message-passing comparisons |
| `lj_ae_08_bridge` | Coordinate-selected experiments close to notebook 08's architecture |
| `compact_lj_motion_audit` | Descriptive trajectory motion and per-network PCA diagnostics |
| `compact_lj_fitted_codes` | Frozen-decoder, per-example latent optimization diagnostics |
| `compact_lj_spatial_residuals` | Post-fit spatial decomposition of reconstruction residuals |
| `compact_lj_spatial_decoder` | Spatial-decoder experiment snapshot; some runs were pending when copied |
| `in_memory_normalization_audit` | Coordinate-normalization checks |
| `latent_*`, `06b_*`, `lj_ae_capacity`, `lj_ae_repair`, `shared_ae_architecture`, `ae_information_audit` | Earlier comparisons, controls, and diagnostic attempts |

The [results index](../notebooks/latent_space/experiment_results_index.md) and linked experiment logs provide the interpretation, exact recipes, and known confounds. Directory names preserve historical experiment identities; they do not imply that every recipe is a current baseline. In particular, response-selected runs are historical evidence, not valid state-only model-selection recipes.

This is a standalone snapshot, not a live view of cluster jobs. Running or queued jobs retain their recorded status until results are collected. Historical manifests may contain the original machine paths; these are provenance records and are left unchanged. The portable HPC runners resolve dataset names against this project's `data/` directory for new work. New runs should use the output location documented in `hpc/README.md`.

Exact notebook-specific cache directories for several 06–10 notebooks were unavailable. Preserved notebook plots are historical outputs, not fresh executions; see [the notebook map](notebooks.md) for their dependencies. Scheduler chatter, duplicate compressed/normalized data, and the external engineering simulation workspace are omitted. Selected downstream engineering results are in `examples/network_design/`.

`notebooks/results/artifact-manifest.json` records hashes of the copied runtime source and historical result files, together with the donor commit. The donor worktree contained changes, so the file hashes identify the actual copied code. `data/manifest.json` independently identifies the supplied datasets. `environment-tested.json` records the dependency versions used for local validation; it is an environment record, not a cross-platform lockfile.
