"""Static-graph encoder for prescribed-loading global box response."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from global_context_simulator import MeshMessageBlock, _mlp


class StaticGlobalBoxResponse(nn.Module):
    """Predict total yz box displacement from a reference graph and x loading only."""

    def __init__(self, *, pooling: str, hidden_dim: int = 64, attention_heads: int = 4, attention_layers: int = 2, atom_type_count: int = 2, atom_embedding_dim: int = 8):
        super().__init__()
        if pooling not in {"mean", "token"}:
            raise ValueError("pooling must be 'mean' or 'token'.")
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        self.pooling = pooling
        self.atom_embedding = nn.Embedding(atom_type_count, atom_embedding_dim)
        self.node_encoder = nn.Sequential(nn.Linear(3 + atom_embedding_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.blocks = nn.ModuleList(MeshMessageBlock(hidden_dim) for _ in range(2))
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=attention_heads, dim_feedforward=2 * hidden_dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=attention_layers, norm=nn.LayerNorm(hidden_dim))
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pooling == "token" else None
        self.decoder = _mlp(hidden_dim + 5, hidden_dim, 2)

    def encode(self, reference_fractional: Tensor, atom_types: Tensor, edge_features: Tensor, edge_index: Tensor) -> Tensor:
        nodes = self.node_encoder(torch.cat((reference_fractional, self.atom_embedding(atom_types)), dim=-1))
        edges = self.edge_encoder(edge_features)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, edge_index)
        if self.global_token is None:
            return self.transformer(nodes.unsqueeze(0)).squeeze(0).mean(dim=0)
        token = self.global_token.expand(1, -1, -1)
        encoded = self.transformer(torch.cat((token, nodes.unsqueeze(0)), dim=1))
        return encoded[0, 0]

    def decode(self, graph_embedding: Tensor, loading_context: Tensor) -> Tensor:
        if loading_context.ndim != 2 or loading_context.size(-1) != 5:
            raise ValueError("loading_context must have shape [targets, 5].")
        return self.decoder(torch.cat((graph_embedding.expand(loading_context.size(0), -1), loading_context), dim=-1))

    def forward(self, reference_fractional: Tensor, atom_types: Tensor, edge_features: Tensor, edge_index: Tensor, loading_context: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.encode(reference_fractional, atom_types, edge_features, edge_index)
        return self.decode(embedding, loading_context), embedding
