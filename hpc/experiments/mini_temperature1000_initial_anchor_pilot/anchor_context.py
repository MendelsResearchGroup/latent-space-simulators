"""Install the one-variable initial-latent conditioning pilot."""

from __future__ import annotations

import torch
from torch import nn


class InitialLatentDeltaMLP(nn.Module):
    """Baseline delta MLP plus the normalized frame-zero latent."""

    predicts_delta = True

    def __init__(self, latent_dim, hidden_size, *, static_context_dim,
                 graph_context_dim, context_include_temperature=False,
                 context_pool_mode="mean"):
        from lss.dynamics.propagator import _make_context_projection

        super().__init__()
        self.static_context_dim = int(static_context_dim)
        self.context_projection = _make_context_projection(
            self.static_context_dim,
            graph_context_dim,
            context_pool_mode,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim + int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * int(latent_dim) + self.context_projection.output_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, context=None, *, context_is_encoded=False):
        if context is None or context.size(-1) != self.static_context_dim + z.size(-1):
            raise ValueError("Initial-anchor context must be [static context, frame-zero latent].")
        static, initial_z = context.split((self.static_context_dim, z.size(-1)), dim=-1)
        encoded_static = static if context_is_encoded else self.context_projection(static)
        return self.net(torch.cat((z, encoded_static, initial_z), dim=-1))


def install() -> None:
    import lss.dynamics.experiment as experiment
    import lss.dynamics.propagator as propagator
    import lss.dynamics.training as training

    if getattr(training.encode_reference_context, "_initial_anchor_patch", False):
        return
    original = training.encode_reference_context

    def anchored(ae_model, sim, *, pos_dim, normalizers, device, **kwargs):
        static = original(
            ae_model,
            sim,
            pos_dim=pos_dim,
            normalizers=normalizers,
            device=device,
            **kwargs,
        )
        initial_z = training.encode_frame_latent(
            ae_model,
            sim,
            0,
            pos_dim=pos_dim,
            node_feature_mode="normalized_delta",
            normalizers=normalizers,
            device=device,
        ).detach()
        return torch.cat((static, initial_z), dim=-1)

    anchored._initial_anchor_patch = True
    training.encode_reference_context = anchored
    experiment.encode_reference_context = anchored

    original_factory = propagator.make_latent_propagator

    def factory(latent_dim, hidden_size, *, model_type=None, context_dim=0,
                graph_context_dim=None, context_include_temperature=False,
                context_pool_mode="mean", **kwargs):
        if str(model_type).lower() == "delta_initial_latent_mlp":
            return InitialLatentDeltaMLP(
                latent_dim,
                hidden_size,
                static_context_dim=context_dim,
                graph_context_dim=graph_context_dim,
                context_include_temperature=context_include_temperature,
                context_pool_mode=context_pool_mode,
            )
        return original_factory(
            latent_dim,
            hidden_size,
            model_type=model_type,
            context_dim=context_dim,
            graph_context_dim=graph_context_dim,
            context_include_temperature=context_include_temperature,
            context_pool_mode=context_pool_mode,
            **kwargs,
        )

    propagator.make_latent_propagator = factory
    training.make_latent_propagator = factory
    experiment.make_latent_propagator = factory
