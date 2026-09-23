# Project structure

| Area | Reusable code | Notebooks |
| --- | --- | --- |
| Dynamics | `src/lss/dynamics/` | `notebooks/dynamics/` |
| Engineering | `src/lss/engineering/` | `notebooks/engineering/` |
| Historical work | `src/lss/past_experiments/` | `notebooks/past_experiments/` |

## Dynamics

`canonical.py` defines the one recommended architecture: the compact
reference-16 message-passing autoencoder and the one-step latent-delta
propagator. `training.py` and `experiment.py` retain stable import paths for
established recipes; their highly configurable implementations are isolated in
`dynamics/_compat/`. `evaluation.py` exposes reconstruction and autonomous
rollout evaluation, and `analysis.py` holds post-fit analysis.

Completed alternative propagator families live under
`past_experiments/dynamics/`. Tiny modules at their former paths exist only so
recorded runners and checkpoints remain reproducible. New work should not add
another architecture switch to the compatibility workflow without first
showing that the canonical pair cannot answer the scientific question.

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

## Maintenance boundaries

Notebooks are edited directly. Repeated plot logic belongs in `src/lss/plotting.py`
(for example `plot_latent_coordinates`, shared by the two notebook01 studies).
Retired notebook generators and patchers are plain-text records under
`tools/past_experiments/notebook_writers/`; do not regenerate current notebooks
from them. Dataset, estimator, horizon and trajectory selections stay explicit
in notebook cells.

`dynamics/position_evaluation.py` owns the shared fixed-boundary physical AE and
rollout evaluator. `dynamics/response_metrics.py` owns finite-pair scalar scoring.
An endpoint p-ratio and a trajectory-fitted p-ratio remain different estimators;
sharing their score calculation does not make them interchangeable.

`hpc/submission.py` owns the reusable PBS submission primitive;
`hpc/submit.py` owns the catalog workflow, hashing and submission ledger. From
zeus-post, submit through normal zeus. Reuse these helpers for new runners,
never patch a running snapshot.

Compatibility is retained only for concrete active dependencies: the training
and experiment import entry points still serve current notebook/checkpoint
workflows, and four local `evaluate.py` entry points serve their study collectors
and snapshot recipes. They delegate to one maintained implementation. Do not
add similar shims for retired experiments; reconstruct those from recorded
recipes/logs and preserved artifacts when needed. Frozen snapshots remain
unchanged, not synchronized with the active implementation.
