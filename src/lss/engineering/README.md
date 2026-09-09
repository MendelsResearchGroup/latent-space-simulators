# Network engineering

`model.py` loads a frozen AE and its existing response-calibrated latent readout, plus an original network. `geometry.py` maps editable positions and stiffnesses into model inputs while enforcing the declared geometric constraints. `optimization.py` minimizes the frozen readout with joint Adam.

`evaluation.py` freezes final graph files and their hashes before athermal compression verification. `animation.py` renders retained original and optimized trajectories. Thermal studies keep their separately recorded physical protocol in the study runners; do not evaluate thermal networks with the athermal Reid example and call that mixed-temperature validation.

See `notebooks/engineering/02_engineer_one_network.ipynb` for the end-to-end known Reid example and `01_adam_engineering.ipynb` for full-cohort results. The optimizer does not call a simulator. The latent readout is a downstream response proxy, not an autonomous prediction of positions.
