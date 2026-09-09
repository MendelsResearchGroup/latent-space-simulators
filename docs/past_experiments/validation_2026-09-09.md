# Local validation

The standalone copy was checked on 2026-09-08 with the versions recorded in `environment-tested.json`.

- Runtime source and 5,921 source/result files matched their originals by SHA-256; data and companion metadata were checked separately (eight files).
- All 16 notebooks parsed and their 136 code cells passed IPython-transformed syntax compilation.
- The compact-LJ diagnostics reader (formerly notebook 11, now `notebooks/dynamics/diagnostics/compact_lj_reconstruction.ipynb`) executed from the standalone notebook directory: five code cells, eight rich outputs, no errors, and `lss` imported from the standalone source tree. Historical notebook training cells were not rerun.
- The active test suite passed: 66 tests and six subtests. Stale normalization tests were updated to the current per-axis reference-box/raw-stiffness convention. Removed expert-loss and external engineering-pipeline tests are retained under `tests/archive/` with explanations.
- HPC runner help, PBS syntax, recipe paths, collector isolation, and four dry-run matrices passed (30/50/20/40 cells). No jobs were submitted; details are in `hpc/validation.json`.
- A wheel built without resolving new dependencies, installed into an independent temporary directory, and imported successfully from outside the project.

These checks used the existing Python environment. They do not establish a fresh dependency installation on another operating system or a new cluster training result. The available historical caches and missing legacy datasets are documented in `notebooks.md`.

## Notebook cleanup verification

After deleting nine superseded notebooks and renumbering the main sequence, all seven remaining notebooks parsed and their 62 code cells passed IPython-transformed syntax compilation. All four renamed latent-space notebooks retained identical SHA-256 hashes, including saved outputs. Updated navigation links resolved, eight data files matched their recorded sizes, and `git diff --check` passed. Training was not rerun.

The full `tools/check_project.py` command could not complete in the current shell: the default Python lacks `graph_utils`, and `.venv/bin/python` lacks `numpy`. The notebook, link, hash, and asset checks above ran independently; they do not constitute a fresh runtime-import or test-suite pass.

## Post-fit p-ratio evaluation verification

The 2026-09-08 evaluation used the existing environment at `/rg/mendels_prj/alexander.z/DL-course-project/.venv/bin/python`, with this repository's source. All 70 frozen-checkpoint PBS evaluations completed with exit status zero. Collection verified checkpoint hashes, exact validation IDs, all five requested frames, coordinate parity with original results, and physical p-ratio targets against independently loaded raw data. All 23,500 rows were finite; no training or final-test evaluation was performed. See the [full review](../../notebooks/results/compact_lj_pratio/review.md) for recipes and numerical tolerances.

Seventeen targeted estimator, pipeline-contract, and edge tests passed. The refreshed diagnostic notebook executed all six code cells and saved inline outputs. `tools/check_project.py` passed its local import, seven-notebook (63 code cells), and eight-data-file checks using the explicit environment above. This resolves the runtime check for that environment, without claiming a fresh installation.

## Dynamics / engineering reorganization — 2026-09-09

Moved active implementations to `lss.dynamics`, separated AE and propagator architectures, and retained legacy import aliases for saved checkpoints. Retired CV/full-space, peptide, Hessian, and older inverse-design implementations now live under `lss.past_experiments`. Canonical and historical class/pickle identities were checked with the existing donor environment and this repository's source.

The full runnable pytest suite passed **64 tests and six subtests** after the move and validation-loss default change. Three historical tests requiring the absent `scripts.internal_09_transformer_search` module were preserved in `tests/archive/` with an explanation; the independent legacy model test remains runnable under `tests/past_experiments/`. The initial unittest-discovery attempt exposed that missing script and was not a passing full-suite run.

The propagator default now selects `val_loss`; cache version 4 includes this default so historical implicit response-selected caches cannot match the new default. Existing explicit historical configurations and frozen result snapshots remain untouched. Cache tests cover the state-loss default and distinct response-selected recipes.

Moved notebook schemas, transformed code-cell syntax, imports, data-file presence/sizes, and navigation links were checked. The seven existing showcase GIFs retain all 121/200 frames and 4,000 ms playback after moving alongside the engineering notebook. The exact joint Adam extraction matched the donor implementation on pristine Reid 84 for two steps, including identical final edit tensors; initial score 0.043594829738140106 and successive scores 0.026133479550480843, 0.007926616817712784 for the parity configuration (learning rate 0.02, stiffness bound 0.25, two steps, blocking-node masking enabled; other optimizer defaults unchanged).

Comparison with the earlier cohort CSV is approximate, not bit-for-bit full-run parity: historical Reid 84 used 555 Adam steps, predicted score -1.0304474830627441 and physical p-ratio -0.1890635266367446. The reorganized end-to-end run used 594 steps and physical p-ratio -0.1892959352030621, a difference of -0.0002324085663174824; original p-ratios agree within 5.6e-17. The separate two-step learning-rate-0.02 extraction test is exact. The full-run discrepancy is retained rather than tuning against LAMMPS to eliminate it.
The notebook fixes Torch CPU threads to 1; the historical sweep runner fixed them to 4 (`hpc/experiments/reid_joint_sweep/run.py`). This is a concrete execution-setting difference and a possible source of floating-point optimizer divergence; causality has not been established. No physical result was used to retune the recipe.

Final notebook execution and persistence audit: the first 594-step draft did not save raw history/edits; an ML-only recovery failed exact graph parity and is recorded as a failure, not attached as the original history. An earlier background kernel retry was subsequently found complete as `runs/engineering/reid84_known_case_20260909T065259Z`; it is a separate full known-case run and is retained. Root executed the final notebook as `runs/engineering/reid84_known_case_20260909T070419Z` after adding history/edits persistence before verification. These two updated executions match byte-for-byte for history.csv and edits.pt, both stop at 565 steps with frozen ML score -1.0302908420562744, and both physically verify original +0.14536385489933665 → optimized -0.18743888736952807. The final notebook and adjacent GIF show this latest result, not the earlier draft. All three original/final pairs are retained: 6/6 valid physical checks, 121 frames each; repeated checks of one known network are not six independent engineered examples. No recipe was retuned from physical results.

All three runs now have hash-verified source snapshots. The final two preserve history and edits with hashes in the pre-verification frozen recipe. The initial kernel socket-permission failure, missing first-draft history, failed exact recovery, and interrupted duplicate ML-only recovery remain documented. Final execution request, notebook source, launcher, all-run metrics and notebook/GIF hashes are under `notebooks/results/engineering_workflow/`. Final checks: four executed notebook code cells, no error outputs, GIF 640×336 / 121 frames / 4,000 ms, exact current-run history/edit persistence, matching current-run repeats.
