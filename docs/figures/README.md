# Architecture overview

See the [short technical guide](architecture_notes.md) for embeddings, attention, and propagation.

Edit `shared_ae_propagator.drawio` in draw.io / diagrams.net. `shared_ae_propagator.svg` is the matching vector preview.

Frame-dependent node embeddings hᵢ combine node features, static reference context, and aggregated adjacent-edge features. Two message-passing rounds update these embeddings before global attention produces z. These are stages of the complete encoder; the funnel labelled Encoder highlights global pooling.

Static per-node embeddings H_ref separately condition the decoder. Their pooled graph context c conditions the propagator. Updated frame-dependent hᵢ embeddings do not bypass the latent bottleneck into the decoder. The propagator predicts a latent increment, which is added to the current latent before decoding; the feedback arrow denotes autonomous rollout.

Implementation: [autoencoder.py](../../src/lss/dynamics/autoencoder.py) and [propagator.py](../../src/lss/dynamics/propagator.py). The upper path shows AE reconstruction. The lower branch shows separately trained latent dynamics with the AE frozen, feeding the same decoder for future positions. Normalization is omitted for clarity. The predicted normalized increment is inverted before addition.

Regenerate the existing files with `python tools/analysis/build_architecture_diagram.py`.

The observed input and decoded output are displacements from the fixed reference: uₜ = xₜ − x_ref and x̂ = x_ref + û. The diagram uses u for displacement and Δz for the propagator’s latent increment.
