# Local validation

The standalone copy was checked on 2026-09-08 with the versions recorded in `environment-tested.json`.

- Runtime source and 5,921 source/result files matched their originals by SHA-256; data and companion metadata were checked separately (eight files).
- All 16 notebooks parsed and their 136 code cells passed IPython-transformed syntax compilation.
- Notebook 11 executed from the standalone notebook directory: five code cells, eight rich outputs, no errors, and `lss` imported from the standalone source tree. Historical notebook training cells were not rerun.
- The active test suite passed: 66 tests and six subtests. Stale normalization tests were updated to the current per-axis reference-box/raw-stiffness convention. Removed expert-loss and external engineering-pipeline tests are retained under `tests/archive/` with explanations.
- HPC runner help, PBS syntax, recipe paths, collector isolation, and four dry-run matrices passed (30/50/20/40 cells). No jobs were submitted; details are in `hpc/validation.json`.
- A wheel built without resolving new dependencies, installed into an independent temporary directory, and imported successfully from outside the project.

These checks used the existing Python environment. They do not establish a fresh dependency installation on another operating system or a new cluster training result. The available historical caches and missing legacy datasets are documented in `notebooks.md`.
