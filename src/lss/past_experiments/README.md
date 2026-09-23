# Past experiments

Retired full-space/CV simulators and historical training, inverse-design,
peptide, Hessian, and matched-GNN workflows are kept here for research history.
They use explicit `lss.past_experiments` imports. Most former top-level import
paths are removed; recorded dynamics variants retain tiny compatibility imports
for frozen runners. This directory does not promise new feature development.

Current workflows use `lss.dynamics` and `lss.engineering`. Shared utilities
remain in `lss.data`, `lss.graph`, `lss.metrics`, `lss.plotting`, and `lss.utils`.
Convert a historical checkpoint to a separate working copy with
`tools/migrations/checkpoint_imports.py` when it is needed.

`dynamics/` contains completed alternative learned-dynamics studies: direct and
factorized simulators, prescribed latent paths, static-graph autoencoding,
trajectory-family matrices, observed-history/acceleration models, local-error
diagnostics, and radical operator/expert/flow variants. Their former
`lss.dynamics.*` modules are compatibility imports only. The active recommended
model pair is defined in `lss.dynamics.canonical`.
