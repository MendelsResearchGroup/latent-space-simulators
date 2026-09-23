"""The single recommended autoencoder and propagator architecture.

Historical architecture matrices remain reproducible through compatibility
modules, but new dynamics work should start here instead of adding another
model-selection branch.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .autoencoder import NodeDeltaMessagePassingReference16AutoEncoder
from .propagator import make_latent_propagator


@dataclass(frozen=True)
class CanonicalDynamicsConfig:
    """Small architecture surface used by the current mini-network studies."""

    latent_dim: int = 2


def make_canonical_autoencoder(
    *,
    pos_dim: int,
    node_feature_dim: int,
    edge_feature_dim: int,
    reconstruction_dim: int,
    config: CanonicalDynamicsConfig = CanonicalDynamicsConfig(),
) -> nn.Module:
    """Build the compact-reference, two-step message-passing autoencoder."""

    return NodeDeltaMessagePassingReference16AutoEncoder(
        pos_dim=int(pos_dim),
        node_feature_dim=int(node_feature_dim),
        edge_dim=int(edge_feature_dim),
        hidden_size=96,
        latent_dim=int(config.latent_dim),
        latent_tokens=32,
        reconstruction_dim=int(reconstruction_dim),
        message_passing_steps=2,
    )


def make_canonical_propagator(
    *,
    config: CanonicalDynamicsConfig = CanonicalDynamicsConfig(),
) -> nn.Module:
    """Build the one-step latent-delta MLP with mean-pooled reference context."""

    return make_latent_propagator(
        int(config.latent_dim),
        64,
        model_type="delta_mlp",
        context_dim=16,
        graph_context_dim=16,
        context_pool_mode="mean",
    )
