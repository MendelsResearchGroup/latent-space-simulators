# Engineering model

`reid_frozen_ae.pt` is the working copy used by the single-network notebook.
It contains the frozen AE and its existing response-calibrated latent readout.
The historical source is unchanged; the adjacent migration record lists both
file hashes and the rewritten import names. Tensor storage is unchanged.

Recreate this local working copy from the historical artifact with:

```bash
python tools/migrations/checkpoint_imports.py \
  notebooks/results/reid_positive10/study_v1/prepared/reid/prepare/prepared.pt \
  models/engineering/reid_frozen_ae.pt
```

The destination must not exist. `validation.json` records the canonical-load
and two-step optimizer check. Weights remain local artifacts excluded from Git.
