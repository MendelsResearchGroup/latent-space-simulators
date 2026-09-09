"""Active code imports concrete canonical modules without compatibility facades."""

import importlib.util

from lss.dynamics.autoencoder import NodeDeltaAttentionAutoEncoder
from lss.dynamics.propagator import LatentDynamicsMLP
from lss.past_experiments.models.attention_pyramid_simulator import (
    AttentionPyramidSimulator,
)


def test_removed_runtime_aliases_and_models_facade_are_not_importable():
    removed_modules = (
        "lss.latent",
        "lss.models",
        "lss.config",
        "lss.runner",
        "lss.training",
        "lss.hessian",
        "lss.matched_gnn",
        "lss.peptide",
        "lss.peptide_tm",
        "lss.inverse_design_barostat",
        "lss.inverse_design_evaluation",
        "lss.dynamics.models",
    )
    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None


def test_canonical_model_classes_import_from_their_concrete_modules():
    assert NodeDeltaAttentionAutoEncoder.__module__ == "lss.dynamics.autoencoder"
    assert LatentDynamicsMLP.__module__ == "lss.dynamics.propagator"
    assert (
        AttentionPyramidSimulator.__module__
        == "lss.past_experiments.models.attention_pyramid_simulator"
    )
