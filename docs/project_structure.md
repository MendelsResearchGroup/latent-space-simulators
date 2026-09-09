# Project structure

| Area | Reusable code | Notebooks |
| --- | --- | --- |
| Dynamics | `src/lss/dynamics/` | `notebooks/dynamics/` |
| Engineering | `src/lss/engineering/` | `notebooks/engineering/` |
| Historical work | `src/lss/past_experiments/` | `notebooks/past_experiments/` |

## Dynamics

`autoencoder.py` and `propagator.py` define the learned models.
`training.py` fits them, `experiment.py` owns splits, recipes, checkpoints, and
result tables, and `evaluation.py` evaluates reconstruction and autonomous
rollouts. `analysis.py` holds post-fit analysis.

Train on positions, trajectories, and graph structure. Select checkpoints with
the declared state-space validation criterion, then evaluate physical p-ratio
R-squared from reconstructed or predicted positions. Report reconstruction and
rollout separately, source-wise and at stated horizons.

## Engineering

`model.py` loads frozen models and original networks. `geometry.py` defines
editable geometry and constraints; `optimization.py` performs gradient-based
optimization; `evaluation.py` freezes designs and verifies them only after
selection; `animation.py` renders comparisons. See
[the engineering goal](research/engineering_goal.md) for the required ML-only
boundary.

## Shared infrastructure and historical records

`src/lss/data.py`, `graph.py`, `metrics.py`, and `plotting.py` are shared
utilities. `hpc/` holds portable launchers and named experiment runners.
`notebooks/results/` retains recipes, source snapshots, metrics, and result
artifacts. The experiment index and logs remain in `notebooks/latent_space/`
because their stable names are provenance references.

Retired source and notebooks are explicitly under `past_experiments`.
Historical documents are under [docs/past_experiments](past_experiments/README.md).
New workflow APIs require concrete inputs and should fail directly for missing
files, keys, or invalid shapes.
