# Validation records

Validation reports are dated records of a specific working tree and
environment; they are not a claim about the current checkout. The preserved
2026-09-08/09 standalone and reorganization validation record is archived at
[past_experiments/validation_2026-09-09.md](past_experiments/validation_2026-09-09.md).

Run validation appropriate to the change being made. Record new scientific
results in the experiment index and the relevant experiment log.

## Current-import cleanup — 2026-09-09

Using the existing donor Python environment with this repository's source:

- 69 tests passed, including canonical-import and checkpoint-conversion tests.
- Removed import paths are no longer importable; current callers use concrete modules.
- The converted engineering bundle reproduces the recorded Reid 84 initial score and two-step Adam result exactly (learning rate 0.02; one CPU thread).
- Dataset files were unchanged. No training or final physics verification was repeated.

The model conversion record and numerical check are in
[`models/engineering/`](../models/engineering/). Historical notebook outputs
remain the recorded executions, not new results from this cleanup.
