"""Normalized message-passing dynamics with frozen learned nonlocal edges."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class NormalizedMessageBlock(nn.Module):
    """Residual edge/node update with mean aggregation and LayerNorm."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_update = _mlp(3 * hidden_dim, hidden_dim, hidden_dim)
        self.node_update = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        edges = self.edge_norm(
            edges + self.edge_update(torch.cat((edges, nodes[source], nodes[target]), dim=-1))
        )
        aggregate = torch.zeros_like(nodes)
        count = torch.zeros((len(nodes), 1), dtype=nodes.dtype, device=nodes.device)
        aggregate.index_add_(0, target, edges)
        count.index_add_(0, target, torch.ones((target.numel(), 1), dtype=nodes.dtype, device=nodes.device))
        nodes = self.node_norm(
            nodes + self.node_update(torch.cat((nodes, aggregate / count.clamp_min(1.0)), dim=-1))
        )
        return nodes, edges


class NormalizedAccelerationGNN(nn.Module):
    """Predict per-node acceleration from three velocities and current displacement."""

    def __init__(self, *, node_dim: int = 8, edge_dim: int = 10, hidden_dim: int = 64, steps: int = 4):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.blocks = nn.ModuleList(NormalizedMessageBlock(hidden_dim) for _ in range(steps))
        self.decoder = _mlp(hidden_dim, hidden_dim, 2)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def make_directed(edge_features: Tensor, undirected_index: Tensor) -> tuple[Tensor, Tensor]:
        reverse = edge_features.clone()
        reverse[:, :4] *= -1
        return torch.cat((edge_features, reverse)), torch.cat((undirected_index, undirected_index.flip(0)), dim=1)

    def forward(self, node_features: Tensor, edge_features: Tensor, undirected_index: Tensor) -> Tensor:
        directed_features, directed_index = self.make_directed(edge_features, undirected_index)
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(directed_features)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, directed_index)
        return self.decoder(nodes)


def canonical_undirected(edge_index: Tensor) -> Tensor:
    """Return sorted unique i<j relations."""

    pairs = torch.sort(edge_index.long(), dim=0).values.t()
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return torch.unique(pairs, dim=0).t().contiguous()


def select_topk_nonphysical_edges(scores: Tensor, base_edges: Tensor, k_per_node: int) -> Tensor:
    """Select a symmetric union of each node's strongest non-neighbors."""

    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be square")
    nodes = scores.shape[0]
    blocked = torch.eye(nodes, dtype=torch.bool, device=scores.device)
    base = canonical_undirected(base_edges).to(scores.device)
    blocked[base[0], base[1]] = True
    blocked[base[1], base[0]] = True
    masked = scores.clone()
    masked[blocked] = -torch.inf
    available = (~blocked).sum(1)
    if bool((available < k_per_node).any()):
        raise ValueError("not enough nonphysical pairs for requested top-k")
    target = torch.topk(masked, k=k_per_node, dim=1).indices
    source = torch.arange(nodes, device=scores.device)[:, None].expand_as(target)
    return canonical_undirected(torch.stack((source.reshape(-1), target.reshape(-1))))


def random_nonphysical_edges(
    node_count: int, base_edges: Tensor, edge_count: int, *, generator: torch.Generator
) -> Tensor:
    """Draw a deterministic size-matched random nonedge control."""

    base = canonical_undirected(base_edges).cpu()
    blocked = {(int(i), int(j)) for i, j in base.t().tolist()}
    candidates = [(i, j) for i in range(node_count) for j in range(i + 1, node_count) if (i, j) not in blocked]
    if edge_count > len(candidates):
        raise ValueError("requested more random edges than available nonedges")
    order = torch.randperm(len(candidates), generator=generator)[:edge_count]
    return torch.tensor([candidates[int(index)] for index in order], dtype=torch.long).t().contiguous()


def geometric_edge_features(
    reference: Tensor,
    current: Tensor,
    edge_index: Tensor,
    stiffness: Tensor,
    base_indicator: Tensor,
    learned_indicator: Tensor,
) -> Tensor:
    """Reference/current geometry and relation identity for undirected edges."""

    source, target = edge_index
    reference_vector = reference[target] - reference[source]
    current_vector = current[target] - current[source]
    reference_length = torch.linalg.vector_norm(reference_vector, dim=-1, keepdim=True)
    current_length = torch.linalg.vector_norm(current_vector, dim=-1, keepdim=True)
    return torch.cat((
        reference_vector,
        current_vector,
        reference_length,
        current_length,
        current_length - reference_length,
        stiffness.reshape(-1, 1),
        base_indicator.reshape(-1, 1),
        learned_indicator.reshape(-1, 1),
    ), dim=-1)
