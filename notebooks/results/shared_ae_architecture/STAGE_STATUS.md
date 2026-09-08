# Shared AE architecture study status

- Frozen historical controls retain their original edge reversal behavior.
- The isolated orientation-corrected attention control and nonlinear message-passing AE use raw-coordinate-consistent reversal after normalization.
- First smoke submissions `4670279` and `4670280` failed before training: nested AE configuration used `autoencoder_model` instead of `model`; their incomplete directories and logs are preserved.
- Retry smokes `4670281` and `4670282` use frozen `code_v2` and both failed before training from a normalization-offset buffer shape mismatch. A `code_v3` retry (`4670283`, `4670284`) then failed before loading data because it reused the existing attempt-2 directories. These failures and their logs are retained.
- `code_v4` (`4670352`, `4670353`) reached two-epoch training successfully. The orientation-corrected run then failed during post-fit transfer evaluation because its runner assumed the smoke must reduce mixed-T from its fixed 20 held-out evaluation trajectories to two. This is a runner assertion defect, not a training or model failure; MP2 uses the same code path. Artifacts and logs are preserved.
- `code_v5` retains version-specific run directories and the normalization-offset correction, and fixes the mixed-T assertion. The d=2, seed-123 orientation-corrected and MP2 smokes still gate broad submission; they use two epochs, 2 train/2 validation networks per retained source, and evaluate the unchanged 20 mixed-T trajectories post-fit, with shared 8 CPU/32 GB placement.

## Current execution state (2026-09-07)

- Gate smokes passed: orientation-corrected `4670355` (265.0 s, 172,036 parameters) and MP2 `4670356` (388.2 s, 302,212 parameters). Each has `completed.json`, 480 source/frame validation rows (2 retained-source trajectories/source plus 20 mixed-T transfer trajectories, each at six frames), and a source summary. Mixed-T was evaluation-only and final test was not loaded.
- Full `code_v5` matrix is submitted: orientation-corrected, MP2, MP4, and single-stage pooling × D=2/8 × seeds 3456456/123/456/786/2026 (40 declared cells). `jobs.jsonl` records all PBS IDs. A launcher overlap created 26 duplicate PBS attempts; later IDs were cancelled where still queued, while already-terminal duplicate attempts are retained in the ledger as failure artifacts. The collector de-duplicates by first declared cell entry, so retries cannot make a successful cell appear incomplete.
- Collectors `4670425` and `4670426` use `afterany` dependencies. The collector writes `matrix_status_code_v5.csv`, `source_summary_code_v5.csv`, and `collection_code_v5.json`; it reports a matrix complete only when every one of the 40 declared first-attempt cells has both `completed.json` and a source summary.
- Frozen-checkpoint diagnostic `4670357` completed. Its coordinate-only artifact is `notebooks/results/ae_information_audit/frozen_code_diagnostic.csv` (six frames; three trajectories/source in train and validation; encoded, wrong-time, same-source permutation, source/time-mean, and three fitted-code starts). The fixed 100-step code fitting gave only modest best-start normalized coordinate-MSE reductions, so it does not yet support a large encoder-only bottleneck claim.
- Broader matrix is staged pending both retry smoke completions: orientation-corrected, message depth 2, message depth 4, and single-stage pooling; d=2 and d=8; five paired seeds. Mixed-T stays evaluation-only; final test is untouched.

Next commands after successful retry smoke:

```bash
.venv/bin/python scripts/submit_shared_ae_architecture.py
```
