# Local validation

The standalone copy was checked on 2026-09-08 with the versions recorded in `environment-tested.json`.

- Runtime source and 5,921 source/result files matched their originals by SHA-256; data and companion metadata were checked separately (eight files).
- All 16 notebooks parsed and their 136 code cells passed IPython-transformed syntax compilation.
- The compact-LJ diagnostics reader (formerly notebook 11, now `notebooks/latent_space/diagnostics/compact_lj_reconstruction.ipynb`) executed from the standalone notebook directory: five code cells, eight rich outputs, no errors, and `lss` imported from the standalone source tree. Historical notebook training cells were not rerun.
- The active test suite passed: 66 tests and six subtests. Stale normalization tests were updated to the current per-axis reference-box/raw-stiffness convention. Removed expert-loss and external engineering-pipeline tests are retained under `tests/archive/` with explanations.
- HPC runner help, PBS syntax, recipe paths, collector isolation, and four dry-run matrices passed (30/50/20/40 cells). No jobs were submitted; details are in `hpc/validation.json`.
- A wheel built without resolving new dependencies, installed into an independent temporary directory, and imported successfully from outside the project.

These checks used the existing Python environment. They do not establish a fresh dependency installation on another operating system or a new cluster training result. The available historical caches and missing legacy datasets are documented in `notebooks.md`.

## Notebook cleanup verification

After deleting nine superseded notebooks and renumbering the main sequence, all seven remaining notebooks parsed and their 62 code cells passed IPython-transformed syntax compilation. All four renamed latent-space notebooks retained identical SHA-256 hashes, including saved outputs. Updated navigation links resolved, eight data files matched their recorded sizes, and `git diff --check` passed. Training was not rerun.

The full `tools/check_project.py` command could not complete in the current shell: the default Python lacks `graph_utils`, and `.venv/bin/python` lacks `numpy`. The notebook, link, hash, and asset checks above ran independently; they do not constitute a fresh runtime-import or test-suite pass.
