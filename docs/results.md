# Results and provenance

`notebooks/results/` stores split manifests, frozen recipes and source
snapshots, source-wise tables, diagnostics, job records, and model artifacts.
Its directory names preserve experiment identity; they do not make a recipe a
current baseline.

Start with the [experiment results index](../notebooks/latent_space/experiment_results_index.md)
for conclusions, exact configurations, seeds, valid/total counts, and links to
source artifacts. The [research status](research-status.md) provides only the
current high-level interpretation.

Response-selected experiments, simulator-guided engineering, and incomplete
matrices are retained as historical evidence. They must be identified as such
and cannot establish current state-only dynamics or ML-only engineering claims.

Artifacts are a snapshot, not a scheduler view. Result manifests and dataset
manifests identify copied files and inputs; use `hpc/README.md` for new-run
output locations.
