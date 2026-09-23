"""Local-global-local graph simulator for small deforming networks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class MeshMessageBlock(nn.Module):
    """Residual edge and node update on an explicitly directed bond list."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_update = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_update = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        updated_edges = self.edge_norm(
            edges + self.edge_update(torch.cat((edges, nodes[source], nodes[target]), dim=-1))
        )
        aggregate = torch.zeros_like(nodes)
        counts = torch.zeros((nodes.size(0), 1), dtype=nodes.dtype, device=nodes.device)
        aggregate.index_add_(0, target, updated_edges)
        counts.index_add_(0, target, torch.ones((target.numel(), 1), dtype=nodes.dtype, device=nodes.device))
        mean_message = aggregate / counts.clamp_min(1.0)
        updated_nodes = self.node_norm(
            nodes + self.node_update(torch.cat((nodes, mean_message), dim=-1))
        )
        return updated_nodes, updated_edges


class LocalGlobalLocalSimulator(nn.Module):
    """Two local updates, optional global attention, then two local refinements."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        atom_type_count: int,
        hidden_dim: int = 64,
        atom_embedding_dim: int = 8,
        message_steps_each_side: int = 2,
        attention_layers: int = 2,
        attention_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        self.attention_layers = int(attention_layers)
        self.atom_embedding = nn.Embedding(int(atom_type_count), int(atom_embedding_dim))
        self.node_encoder = nn.Sequential(
            nn.Linear(int(node_feature_dim) + int(atom_embedding_dim), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(int(edge_feature_dim), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pre_blocks = nn.ModuleList(
            MeshMessageBlock(hidden_dim) for _ in range(int(message_steps_each_side))
        )
        if self.attention_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(attention_heads),
                dim_feedforward=2 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.global_processor: nn.Module | None = nn.TransformerEncoder(
                layer, num_layers=self.attention_layers, norm=nn.LayerNorm(hidden_dim)
            )
        else:
            self.global_processor = None
        self.post_blocks = nn.ModuleList(
            MeshMessageBlock(hidden_dim) for _ in range(int(message_steps_each_side))
        )
        self.node_decoder = _mlp(hidden_dim, hidden_dim, 3)
        self.box_decoder = _mlp(hidden_dim, hidden_dim, 3)
        nn.init.zeros_(self.node_decoder[-1].weight)
        nn.init.zeros_(self.node_decoder[-1].bias)
        nn.init.zeros_(self.box_decoder[-1].weight)
        nn.init.zeros_(self.box_decoder[-1].bias)

    @staticmethod
    def bidirectional(edge_attr: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        reverse = edge_attr.clone()
        reverse[:, :3] *= -1
        return torch.cat((edge_attr, reverse), dim=0), torch.cat((edge_index, edge_index.flip(0)), dim=1)

    def forward(
        self,
        node_features: Tensor,
        atom_types: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if node_features.ndim != 2 or edge_features.ndim != 2 or edge_index.ndim != 2:
            raise ValueError("Expected unbatched node and edge tensors.")
        if atom_types.ndim != 1 or atom_types.numel() != node_features.size(0):
            raise ValueError("atom_types must have one entry per node.")
        directed_edges, directed_index = self.bidirectional(edge_features, edge_index)
        nodes = self.node_encoder(torch.cat((node_features, self.atom_embedding(atom_types)), dim=-1))
        edges = self.edge_encoder(directed_edges)
        for block in self.pre_blocks:
            nodes, edges = block(nodes, edges, directed_index)
        if self.global_processor is not None:
            # TransformerEncoderLayer already contains residual connections.
            nodes = self.global_processor(nodes.unsqueeze(0)).squeeze(0)
        for block in self.post_blocks:
            nodes, edges = block(nodes, edges, directed_index)
        return self.node_decoder(nodes), self.box_decoder(nodes.mean(dim=0))
