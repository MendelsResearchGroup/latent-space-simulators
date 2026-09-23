# Past experiments

Retired full-space/CV simulators and historical training, inverse-design,
peptide, Hessian, and matched-GNN workflows are kept here for research history.
They use explicit `lss.past_experiments` imports. Former top-level import paths
are removed; this directory does not promise continued support for old recipes.

Current workflows use `lss.dynamics` and `lss.engineering`. Shared utilities
remain in `lss.data`, `lss.graph`, `lss.metrics`, `lss.plotting`, and `lss.utils`.
Convert a historical checkpoint to a separate working copy with
`tools/migrations/checkpoint_imports.py` when it is needed.
