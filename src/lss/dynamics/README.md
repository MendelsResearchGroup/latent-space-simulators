# AE and propagator

- `canonical.py`: the recommended reference-16 message-passing AE and one-step
  delta propagator. Start new work here.
- `autoencoder.py` and `propagator.py`: concrete implementations retained for
  checkpoints and controlled comparisons; they are not a menu of defaults.
- `training.py` and `experiment.py`: stable compatibility imports for recorded
  recipes. Their configurable implementations are isolated in `_compat/`.
- `evaluation.py`: physical p-ratio R² from AE reconstruction and autonomous rollouts.
- `analysis.py`: post-fit latent and response analysis.

Use `notebooks/dynamics/` for the visual workflows. Completed alternative
simulators and propagators live in `lss.past_experiments.dynamics`; tiny files
at their former paths preserve old imports. Read `docs/project_structure.md`
and the results index before choosing a new experiment.

The files `cold_graphs.py`, `global_context_simulator.py`,
`fractional_contact_simulator.py`, `static_response_head.py`,
`initial_motion_response.py`, and `node_only_global_simulator.py` belong only
to the paused 3D cold-rubber side project. They are not exported from this
package and are not part of the canonical 200/1,000-MD-step workflow.
