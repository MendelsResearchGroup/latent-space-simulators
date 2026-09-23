"""Fractional-coordinate local-global-local simulator with dynamic contacts."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lss.dynamics.global_context_simulator import MeshMessageBlock, _mlp


def fractional_coordinates(positions: Tensor, box_lengths: Tensor) -> Tensor:
    return positions / box_lengths


def minimum_image(delta: Tensor, box_lengths: Tensor) -> Tensor:
    return delta - torch.round(delta / box_lengths) * box_lengths


def unit_minimum_image(delta: Tensor) -> Tensor:
    return delta - torch.round(delta)


def build_interaction_graph(
    positions: Tensor,
    bond_index: Tensor,
    box_lengths: Tensor,
    bond_stiffness: Tensor,
    *,
    contact_cutoff: float | None,
) -> tuple[Tensor, Tensor]:
    """Return directed bond edges plus optional dynamic nonbonded contacts.

    Features are minimum-image displacement (3), distance, bond indicator, and
    bond stiffness. Each physical pair appears in both directions exactly once.
    """
    source, target = bond_index
    bond_delta = minimum_image(positions[source] - positions[target], box_lengths)
    bond_length = torch.linalg.vector_norm(bond_delta, dim=-1, keepdim=True)
    stiffness = bond_stiffness.reshape(-1, 1)
    forward = torch.cat((bond_delta, bond_length, torch.ones_like(stiffness), stiffness), dim=-1)
    reverse = forward.clone()
    reverse[:, :3] *= -1
    edge_index = torch.cat((bond_index, bond_index.flip(0)), dim=1)
    edge_features = torch.cat((forward, reverse), dim=0)
    if contact_cutoff is None:
        return edge_features, edge_index

    pair_delta = positions[:, None, :] - positions[None, :, :]
    pair_delta = minimum_image(pair_delta, box_lengths)
    pair_distance = torch.linalg.vector_norm(pair_delta, dim=-1)
    mask = (pair_distance > 1e-8) & (pair_distance < float(contact_cutoff))
    mask[source, target] = False
    mask[target, source] = False
    contact_source, contact_target = torch.nonzero(mask, as_tuple=True)
    contact_delta = pair_delta[contact_source, contact_target]
    contact_length = pair_distance[contact_source, contact_target, None]
    zeros = torch.zeros_like(contact_length)
    contact_features = torch.cat((contact_delta, contact_length, zeros, zeros), dim=-1)
    contact_index = torch.stack((contact_source, contact_target))
    return torch.cat((edge_features, contact_features), dim=0), torch.cat((edge_index, contact_index), dim=1)


class FractionalContactSimulator(nn.Module):
    """Local-global-local dynamics in a deforming fractional coordinate frame."""

    def __init__(
        self,
        *,
        node_feature_dim: int = 9,
        edge_feature_dim: int = 6,
        global_feature_dim: int = 9,
        atom_type_count: int = 2,
        hidden_dim: int = 64,
        atom_embedding_dim: int = 8,
        attention_heads: int = 4,
        attention_layers: int = 2,
        message_steps_each_side: int = 2,
        recurrent: bool = False,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        self.recurrent = bool(recurrent)
        self.atom_embedding = nn.Embedding(atom_type_count, atom_embedding_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim + atom_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.recurrent_cell = nn.GRUCell(hidden_dim, hidden_dim) if self.recurrent else None
        self.pre_blocks = nn.ModuleList(MeshMessageBlock(hidden_dim) for _ in range(message_steps_each_side))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_processor = nn.TransformerEncoder(
            layer,
            num_layers=attention_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.post_blocks = nn.ModuleList(MeshMessageBlock(hidden_dim) for _ in range(message_steps_each_side))
        self.node_decoder = _mlp(hidden_dim, hidden_dim, 3)
        self.box_decoder = _mlp(2 * hidden_dim, hidden_dim, 2)
        nn.init.zeros_(self.node_decoder[-1].weight)
        nn.init.zeros_(self.node_decoder[-1].bias)
        nn.init.zeros_(self.box_decoder[-1].weight)
        nn.init.zeros_(self.box_decoder[-1].bias)

    def forward(
        self,
        node_features: Tensor,
        atom_types: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        global_features: Tensor,
        hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        global_embedding = self.global_encoder(global_features)
        observed = self.node_encoder(torch.cat((node_features, self.atom_embedding(atom_types)), dim=-1))
        observed = observed + global_embedding
        if self.recurrent_cell is not None:
            if hidden_state is None:
                hidden_state = torch.zeros_like(observed)
            nodes = self.recurrent_cell(observed, hidden_state)
        elif hidden_state is not None:
            raise ValueError("hidden_state is only valid for a recurrent simulator.")
        else:
            nodes = observed
        edges = self.edge_encoder(edge_features)
        for block in self.pre_blocks:
            nodes, edges = block(nodes, edges, edge_index)
        nodes = self.global_processor(nodes.unsqueeze(0)).squeeze(0)
        for block in self.post_blocks:
            nodes, edges = block(nodes, edges, edge_index)
        next_hidden = nodes if self.recurrent else None
        box_context = torch.cat((nodes.mean(dim=0), global_embedding), dim=-1)
        return self.node_decoder(nodes), self.box_decoder(box_context), next_hidden
