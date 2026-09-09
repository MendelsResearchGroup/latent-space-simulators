# Models

Active learned dynamics lives in `lss.dynamics`. The package learns from node
positions, trajectories, and graph structure. P-ratio and strain are post-fit
diagnostics, never model inputs, losses, or checkpoint-selection criteria.

## Autoencoder

`lss.dynamics.autoencoder` defines the graph autoencoders and
`lss.dynamics.experiment` selects them for a named recipe. The standard model
encodes reference geometry, current node state, and edge features into a graph
latent, then decodes node coordinates or displacements. Compact-LJ recipes use
`normalized_delta` with reference-box normalization.

The named AE families are attention, orientation-corrected attention,
compact-reference attention, message-passing, direct-attention,
single-stage-attention, and MLP baselines. Compact-reference and
message-passing variants retain a small per-node reference representation while
the dynamic encoder remains wider. The spatial decoder is an experimental
decoder family, not a default.

LJ uses the same stored edge schema during fitting and inference: two edge-vector
components, length, raw spring stiffness, and an LJ-relation indicator. Missing
spring-graph distance-two and distance-three pairs are included with zero raw
stiffness and indicator one.

## Propagator

`lss.dynamics.propagator` and `make_latent_propagator` define latent transition
models. `delta_mlp` is the ordinary state-only baseline: it predicts a one-step
latent increment from the current latent and optional learned graph context.
Other supported families include residual/direct MLPs, kinematic models, causal
history models, recurrent models, and historical controls. A history model has
a larger observed-state budget and must be reported separately from a
current-state model.

The AE reconstructs observed frames. Autonomous prediction requires a fitted
propagator and must be evaluated separately from reconstruction. Report
physical-coordinate p-ratio R-squared source-wise at stated horizons, with
valid/total counts, alongside secondary coordinate diagnostics.

## Engineering and history

`lss.engineering` loads frozen models and original networks, optimizes bounded
edits with model gradients, freezes final designs, and performs final-only
verification. Its workflow and scientific boundary are defined in
[engineering_goal.md](research/engineering_goal.md).

`lss.past_experiments` contains retired implementations retained only for
reproducibility. New code must import `lss.dynamics` or `lss.engineering`;
there is no ongoing legacy import-alias support.

The single-network notebook uses the converted working bundle
`models/engineering/reid_frozen_ae.pt`; its adjacent migration record identifies
the unchanged historical source and the converted-file hashes.

## Historical checkpoints

Trusted PyTorch ZIP checkpoints with retired import names must be converted
once before loading. Run:

```bash
python tools/migrations/checkpoint_imports.py SOURCE DESTINATION
```

The converter rewrites pickle import references, leaves the source checkpoint
unchanged, verifies every non-pickle ZIP member, and writes a hash-backed
`DESTINATION.migration.json` record. It does not promise that every historical
checkpoint is compatible with the current runtime.
