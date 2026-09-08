# Latent-space simulator models

The active work learns dynamics from observed node coordinates/trajectories and graph structure. Coordinate reconstruction and latent-dynamics losses are the operative objectives. P-ratio and strain are post-fit diagnostics; slow-mode labels and physics-informed losses belong to historical training branches and are not inputs, targets, or selection criteria for the active state-only recipes.

## Autoencoders

`src/lss/latent/experiment.py::_autoencoder_class` is the model registry. The standard attention AE builds reference-node features from reference positions and edges, combines current node features with evolving edge features, pools node states into a graph latent, and decodes per-node coordinates/displacements using node queries over learned latent tokens. `normalized_delta` is the current compact-LJ input/target convention, under per-trajectory reference-box coordinate normalization.

- `attention` / `attention_mlp`: baseline pyramid-attention AE.
- `orientation_corrected`: baseline with reversal-consistent edge handling.
- `attention_reference8`, `attention_reference16`, `attention_reference16_corrected`: retain an 8D/16D reference representation per node while dynamic hidden width remains larger. The corrected 16D form is the compact-reference default under current study.
- `message_passing_reference16`: compact 16D reference plus residual nonlinear neighbour-message steps before pooling. `message_passing` applies the same idea to the full reference variant.
- `attention_reference16_spatial_decoder`: compact reference with a graph-wide Transformer decoder over node states after latent-token attention; it is a candidate spatial-pattern decoder, not the established default.
- `direct_attention`: decoder attention directly returns displacement values; `single_stage_attention` pools nodes directly to learned latent queries.
- `mlp` / `pyramid_mlp`: mean-pooled MLP alternatives.

The AE reconstructs a frame from its observed state. It is not an autonomous simulator by itself. Compact-LJ recipes use fixed stored graph edges and five edge channels: vector (2), length, raw spring stiffness, and an LJ-relation indicator. LJ additionally connects missing original-spring-graph distance-2/3 pairs with zero raw stiffness and indicator one. The same schema is required through fitting and inference.

## Latent propagators

`src/lss/latent/models.py::make_latent_propagator` is the transition-model factory. Static graph context is a pooled reference representation; it is not an expert observable. The ordinary active reference-simplification model is `delta_mlp`: current `z` plus optional graph context, predicting a one-step latent delta. It trains on adjacent observed transitions and rolls out autonomously from frame zero.

| Factory names | Inputs and causal budget | Output and use |
| --- | --- | --- |
| `residual_mlp`, `delta_mlp`, `linear` | Current `z`; optional pooled context | Respectively residual next `z`, delta `Δz`, or linear residual/delta transition. `delta_mlp` is the active baseline. |
| `direct_mlp` / `jepa_mlp` | Current `z`; optional context | Direct next latent `z_next`, for next-embedding objectives. |
| `kinematic_mlp`, `anchored_mlp`, `second_order_direct`; `velocity_mlp` | Current latent state in their defined kinematic/velocity parameterization; optional context | Second-order or velocity-style next-state update. |
| `history_mlp`, `history_delta_mlp`, `history_attention` | A causal window of `history_depth` prior/current latents; optional context | Direct next latent, delta, or attention-based next latent. These consume more observed history than current-state models. |
| `fixed_velocity_residual`, `fixed_window_velocity_residual`, `fixed_window` | Current latent plus fixed observed velocity/window inputs; optional progress if enabled | Residual next latent. The fixed window size is configurable. |
| `fixed_history`, `fixed_window_history_context`, `fixed_window_history_gated_context` | Fixed observed latent history; the latter two construct learned or gated motion context | Next/residual latent with an explicit larger observed-history budget. |
| `recurrent_memory_gru` | Current latent and recurrent hidden memory; optional context | GRU-based next latent. |
| `polar`, `polar_rho`, `rho_theta`, `radial` | Latent represented in the model's polar form; optional context | Polar/radial latent transition. |
| `delta_source_classifier`; `source_conditioned_fixed_velocity_residual`; `noisy_residual_fixed_velocity_residual` | Include source labels or specialised fixed-velocity/noise inputs | Specialised historical/control transitions, not active shared state-only recipes. |

`context_include_temperature` and source-name conditioning are factory options but are outside the current state-and-structure-only recipes. Objectives include one-step latent delta/next-state fitting and rollout-coordinate evaluation; active comparisons use source-wise coordinate validation. Historical response-selected, p-ratio-supervised, strain-supervised, and physics branches remain for provenance but are incompatible with the current requirement.

## Separate simulator modules

`static_autoencoder.py::StaticGraphAutoEncoder` is a masked static-graph denoising AE. It message-passes masked node/edge features, produces local node/edge encodings and a pooled static vector, then reconstructs node and edge features.

`factorized_simulator.py` separates a displacement-only dynamic code from static graph features. `ConditionalDynamicAutoEncoder` pools node displacements to a dynamic code and decodes them with static node/global features. `StaticConditionedDeltaPropagator` predicts a residual dynamic-code update; `StaticConditionedNextStepSimulator` re-encodes the current displacement and decodes a coordinate increment; `AttentionStaticConditionedPropagator` applies Transformer attention over dynamic-coordinate tokens conditioned on the static code.

`direct_autoencoder_simulator.py` trains a registered AE directly on observed frame-to-next-frame transitions and provides next-graph prediction and rollout evaluation. It is a direct coordinate-transition route, separate from freezing an AE and fitting a latent propagator.
