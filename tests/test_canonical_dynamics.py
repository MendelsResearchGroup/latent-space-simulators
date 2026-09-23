"""The active dynamics API stays intentionally small and opinionated."""

from lss.dynamics.autoencoder import NodeDeltaMessagePassingReference16AutoEncoder
from lss.dynamics.canonical import (
    CanonicalDynamicsConfig,
    make_canonical_autoencoder,
    make_canonical_propagator,
)
from lss.dynamics.propagator import DeltaLatentDynamicsMLP


def test_canonical_factories_build_the_current_model_pair():
    config = CanonicalDynamicsConfig()
    autoencoder = make_canonical_autoencoder(
        pos_dim=2,
        node_feature_dim=2,
        edge_feature_dim=4,
        reconstruction_dim=2,
        config=config,
    )
    propagator = make_canonical_propagator(config=config)

    assert isinstance(autoencoder, NodeDeltaMessagePassingReference16AutoEncoder)
    assert autoencoder.latent_dim == 2
    assert autoencoder.reference_dim == 16
    assert autoencoder.message_passing_steps == 2
    assert isinstance(propagator, DeltaLatentDynamicsMLP)
    assert propagator.context_projection.raw_context_dim == 16
    assert propagator.context_dim == 16
