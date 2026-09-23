# HPC

```bash
export LSS_PROJECT_ROOT=$PWD
python hpc/submit.py compact_lj_reconstruction --smoke --dry-run
python hpc/submit.py compact_lj_reconstruction --submit-host zeus
```

`--dry-run` writes the planned matrix and frozen snapshot but never calls
`qsub`. Full runs submit through `zeus` by default using its
`/usr/local/bin/qsub`, with `mendels_q`, shared placement, 8 CPUs, 16 GB, and
an `afterany` collector. On a zeus-post, continue using `--submit-host zeus`.
Use `--submit-host local` only when the current machine itself provides the
intended `qsub`. Submission is never retried automatically, to avoid duplicate
jobs after an ambiguous scheduler or network failure. Set `LSS_PYTHON` when
needed.

New jobs use `notebooks/results/hpc_runs/`; historical results stay unchanged.
See `experiments/README.md` for matrices and inputs.
