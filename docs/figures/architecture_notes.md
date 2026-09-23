# Architecture: short technical guide

Companion to the [draw.io chart](shared_ae_propagator.drawio) and [SVG preview](shared_ae_propagator.svg). This describes the Reference16 + two-round message-passing configuration; widths below are configuration-specific.

**What are the representations?**

| Symbol | Shape | Meaning |
|---|---|---|
| H_ref | N × 16 | Static embedding for each reference node |
| hᵢ | 96 per node | Frame-dependent node embedding, updated by message passing |
| c | 16 | Static graph context: mean of H_ref over nodes |
| zₜ | d | Global evolving state, with d = 2 or 4 in the illustrated variants |

H_ref stays available outside the dynamic bottleneck. A 2D z therefore describes the evolving state **conditioned on the retained reference network**.

**How are the static node embeddings built?** Reference positions and mean incident-edge features pass through learned linear projections to give one 16D vector per node. Edge directions are oriented consistently for each endpoint. These vectors stay fixed throughout a trajectory. The decoder gets all of H_ref; the propagator gets its pooled graph summary c.

**How are the dynamic node embeddings built?** For each frame, lift the static node embedding to width 96, add a projected mean of current adjacent-edge features, concatenate the node displacement from reference, and linearly project to hᵢ. Thus each embedding contains node information and local edge information.

**What does message passing do?** Each neighbor j sends node i a learned message from the concatenation of hᵢ, hⱼ, and the connecting edge embedding. Node i averages its incoming messages, applies an update MLP, adds the result to its existing embedding, and applies LayerNorm. Two rounds allow exchange across two hops of the supplied graph. For LJ, those are hops in the augmented model graph. Message passing and the following attention pooling are both parts of the complete encoder.

**What is global attention in the encoder?** Updated node embeddings supply the **keys K and values V**. Learned query vectors supply **Q**: they are trainable parameters, not individual nodes. Each query compares itself with every key, then takes a weighted sum of values:

`attention weights = softmax(Q Kᵀ / √96)`

`output tokens = attention weights × V`

A key determines how relevant a node is to a query; its value supplies the information collected. This is global cross-attention pooling. The implementation reduces **N nodes → 100 tokens → 20 tokens → d scalar latent coordinates**. Each stage has its own learned queries and projections. At the final stage, each latent coordinate has its own query and scalar value channel. The result is zₜ.

**What does the propagator predict?** An MLP takes zₜ and static graph context c and predicts a latent increment Δẑₜ. After undoing the increment normalization:

`ẑₜ₊₁ = zₜ + Δẑₜ`

For autonomous rollout, feed the predicted latent back into the same propagator while keeping c fixed. It predicts latent changes, not positions directly; future observed frames are not supplied.

**How does the decoder use attention?** A learned linear projection expands z into 32 tokens of width 96. These tokens supply **keys and values**. Each static node embedding hᵢ_ref supplies that node's **query**. Attention gives each node its own weighted combination of the latent tokens. An MLP receives this combination together with hᵢ_ref and predicts the node's displacement. Undo normalization and add the reference position to recover physical coordinates. Dynamic message-passing embeddings do not bypass z into the decoder.

**Reconstruction versus prediction:** decoding an encoded zₜ reconstructs the observed frame; decoding a propagated ẑₜ₊₁ predicts a future frame. p-ratio and strain are calculated afterward from positions, not supplied as inputs or training targets.

Implementation: [autoencoder.py](../../src/lss/dynamics/autoencoder.py) and [propagator.py](../../src/lss/dynamics/propagator.py).
