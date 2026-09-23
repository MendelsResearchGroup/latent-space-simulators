
### Completed frozen LJ training-versus-validation response audit — 2026-09-08

Allfive evaluation jobs4674805–4674809 produced successful completion markers,
300 training rows each (60 training LJ networks ×five horizons). No fitting,
checkpoint selection or reserved-data access. Training checkpoints unchanged.
Historical PBS queries were intermittently unavailable; terminal success here
is supported by saved completed.json artifacts, not a claim of verified PBS
exit codes. Exact runner, data, split IDs, checkpoint, metric-module and collector
hashes are saved in compact_lj_training_gap/{jobs_submitted.json,collection.json}.
Collector SHA2563ae10e25a535353092af97cd8c6e919219ed8ece7e9753fa2a4ec6f694894709.

Physical endpoint p-ratio R² mean ±sample SD across seeds3456456/123/456/786/2026.
Every training cell has300/300 valid predictions (60 unique networks repeated
five times); every validation cell100/100 (20 unique networks repeated). Exact
train/validation IDs are disjoint, all five frames present, duplicates rejected.

| Frame | Training R² | Validation R² | Training p MAE | Validation p MAE | Training target variance | Validation target variance |
| --- | --- | --- | --- | --- | --- | --- |
|25|0.381±0.210|0.217±0.305|0.04922|0.04701|0.006743|0.004217|
|50|0.712±0.080|0.605±0.135|0.03988|0.03497|0.009735|0.004872|
|100|0.824±0.025|0.633±0.095|0.03755|0.03604|0.013043|0.005393|
|150|0.840±0.013|0.351±0.308|0.03825|0.04713|0.014937|0.005919|
|199|0.817±0.041|0.127±0.284|0.04222|0.05825|0.017069|0.007100|

Interpretation: late response reconstruction is possible on training networks;
there is a held-out response-error gap. Do not interpret the R² difference
alone as overfitting: validation target variance is smaller (about2.4x smaller
at199), and validation MAE is actually slightly lower at25–100. At150/199,
validation MAE also increases; late generalization is therefore a real concern,
not solely an R² denominator effect. These data motivate testing additional
independent LJ trajectories, but do not prove that more data will fix the gap.
Keep the validation cohort fixed and the120 reserved LJ trajectories untouched.
No additional training was launched by this diagnostic.

Existing MP4D histories selected coordinate checkpoints at epochs8–10 and
stopped20–22 under max60/patience12. Raising max_epochs alone would not bypass
that early stop, and coordinate histories do not establish p-ratio convergence.
The response-rollout notebook now includes inline training/validation R² and
MAE curves plus target-variance and count tables.
