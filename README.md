# Latent-space simulators

Learning shared collective variables and dynamics across ensembles of spring and Lennard–Jones networks. A graph autoencoder maps observed node motion into a small latent space; a propagator evolves that state, and the decoder reconstructs node coordinates.

Start with [notebook 06](notebooks/latent_space/06_mixed_dataset_shared_latent_space.ipynb) for the shared encoding and its relationship to network response. Later notebooks study rollout, transfer between ensembles, and noisy-LJ reconstruction, which remains an open problem.

- [Notebooks](docs/notebooks.md)
- [Models and variants](docs/models.md)
- [Current results](docs/research-status.md)
- [HPC experiments](hpc/README.md)

## Setup

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[notebooks,dev]'
python -m ipykernel install --user --name latent-space-simulators
jupyter lab
```

Choose the `latent-space-simulators` kernel. Installation fetches `graph_utils` from the research group's GitHub repository.

The local copy includes the available data and checkpoints (about 7.2 GB). Large files are ignored by Git, so a future clone will need them supplied separately. Some older notebook checkpoints and two legacy datasets are unavailable; saved notebook figures are preserved. See [data](data/README.md) and [result availability](docs/results.md).

## Working with the project

`src/lss/` contains the models, data handling, training, and evaluation. `notebooks/` contains the research sequence and results; `hpc/` contains runners and PBS scripts. Selected network-design results are in `examples/network_design/`.

```bash
python -m pytest -q
python tools/check_project.py
python hpc/submit.py compact_lj_reconstruction --smoke --dry-run
```

[Validation](docs/validation.md) records what was checked. [Research conventions](CONTRIBUTING.md) cover experiments and notebooks. Active models use observed states and graph structure; p-ratio is evaluated after fitting. Historical response-selected experiments are identified in the notebook map and logs.
