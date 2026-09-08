# Experiment catalog

| Study | Matrix | Required input |
| --- | --- | --- |
| `reference_simplification` | 96/16/8 reference × 2/4 latent × 5 seeds | bridge split manifest (optional historical AE reuse) |
| `compact_lj_reconstruction` | 10 variants × 5 seeds | bridge split manifest |
| `compact_lj_spatial_decoder` | 4 variants × 5 seeds | bridge split manifest |
| `lj_ae_08_bridge` | 8 variants × 5 seeds; no smoke | matching 06b baseline recipe |

Portable runners and helpers are in `runners/`. Each submission freezes them,
`src/`, PBS files, and hashes. Historical manifests remain unchanged; runtime
resolves dataset basenames below `$LSS_PROJECT_ROOT/data`.

Post-fit diagnostics are also in `runners/`: motion, fitted-code, spatial
residual, and compact comparison. Use `$LSS_HISTORICAL_RESULTS_ROOT` for inputs
and `$LSS_RESULTS_ROOT` for fresh output.
