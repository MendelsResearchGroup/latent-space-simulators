# HPC

```bash
export LSS_PROJECT_ROOT=$PWD
python hpc/submit.py compact_lj_reconstruction --smoke --dry-run
python hpc/submit.py compact_lj_reconstruction
```

`--dry-run` writes the planned matrix and frozen snapshot but never calls
`qsub`. Full runs use local `qsub`, `mendels_q`, shared placement, 8 CPUs,
16 GB, and an `afterany` collector. Set `LSS_PYTHON` when needed.

New jobs use `notebooks/results/hpc_runs/`; historical results stay unchanged.
See `experiments/README.md` for matrices and inputs.
