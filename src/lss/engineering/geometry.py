"""Differentiable bounded geometry used by the frozen engineering optimizer."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from graph_utils import directional_side_indices_from_box

EPS = 1e-8


def local_length_scales(reference) -> torch.Tensor:
    edge_index = reference.edge_index.long()
    lengths = reference.reference_context_edge_attr[:, 2]
    return 0.25 * torch.stack(
        [lengths[(edge_index == node).any(0)].min() for node in range(len(reference.x))]
    )


def model_geometry(reference, physical, log_stiffness: torch.Tensor, position_q: torch.Tensor):
    """Make an edited normalized model graph while retaining physical coordinates."""
    edge_index = reference.edge_index.long()
    stiffness0 = reference.edge_attr[:, 3]
    position0 = reference.reference_context_positions[:, :2]
    box = torch.tensor([physical.box.x, physical.box.y], dtype=position0.dtype)
    center = torch.tensor(
        [(physical.box.x1 + physical.box.x2) / 2, (physical.box.y1 + physical.box.y2) / 2],
        dtype=position0.dtype,
    )
    position = position0 + position_q * local_length_scales(reference)[:, None]
    stiffness = stiffness0 * log_stiffness.exp()

    def edge_data(coordinates: torch.Tensor, periodic_box: torch.Tensor) -> torch.Tensor:
        vector = coordinates[edge_index[1]] - coordinates[edge_index[0]]
        vector = vector - torch.round(vector / periodic_box) * periodic_box
        return torch.cat([vector, vector.norm(dim=1, keepdim=True), stiffness[:, None]], dim=1)

    graph = reference.clone()
    normalized = (position - center) / (box / 2)
    graph.x = torch.cat([normalized, reference.x[:, 2:]], dim=1)
    graph.pos = normalized
    graph.edge_attr = edge_data(normalized, torch.tensor([2.0, 2.0], dtype=position0.dtype))
    graph.reference_context_positions = position
    graph.reference_context_edge_attr = edge_data(position, box)
    return graph, position, stiffness, graph.reference_context_edge_attr[:, 2], box


def boundary_constraints(reference, physical):
    position = reference.reference_context_positions[:, :2]
    sides = directional_side_indices_from_box(physical, quantile=0.1)
    fixed = torch.zeros(len(position), dtype=torch.bool)
    for indices in sides.values():
        fixed[torch.as_tensor(indices)] = True
    epsilon = 1e-5 * float(reference.reference_context_edge_attr[:, 2].median())
    lower = torch.stack(
        [position[sides["left"], 0].max() + epsilon, position[sides["bottom"], 1].max() + epsilon]
    )
    upper = torch.stack(
        [position[sides["right"], 0].min() - epsilon, position[sides["top"], 1].min() - epsilon]
    )
    return sides, fixed, lower, upper


def project_log_stiffness_(log_stiffness: torch.Tensor, active: torch.Tensor, bound: float) -> None:
    values = log_stiffness[active].clone()
    lower, upper = float(values.min()) - bound, float(values.max()) + bound
    for _ in range(60):
        midpoint = (lower + upper) / 2
        if float((values - midpoint).clamp(-bound, bound).mean()) > 0:
            lower = midpoint
        else:
            upper = midpoint
    log_stiffness[active] = (values - (lower + upper) / 2).clamp(-bound, bound)
    log_stiffness[~active] = 0


def project_positions_(position_q, position0, scales, fixed, lower, upper, radius: float) -> None:
    position_q.mul_((radius / position_q.norm(dim=1, keepdim=True).clamp_min(EPS)).clamp_max(1))
    coordinates = (position0 + position_q * scales[:, None]).maximum(lower).minimum(upper)
    position_q[:] = (coordinates - position0) / scales[:, None]
    position_q[fixed] = 0


def valid_geometry(reference, physical, log_stiffness, position_q, sides) -> bool:
    _, position, _, lengths, box = model_geometry(reference, physical, log_stiffness, position_q)
    original_lengths = reference.reference_context_edge_attr[:, 2]
    if not torch.isfinite(position).all() or (lengths < 0.25 * original_lengths).any() or (lengths > 1.75 * original_lengths).any():
        return False
    pairs = torch.triu_indices(len(position), len(position), 1)
    distances = position[pairs[1]] - position[pairs[0]]
    distances -= torch.round(distances / box) * box
    original = reference.reference_context_positions[:, :2]
    original_distances = original[pairs[1]] - original[pairs[0]]
    original_distances -= torch.round(original_distances / box) * box
    if distances.norm(dim=1).min() < 0.1 * original_distances.norm(dim=1).min():
        return False
    probe = deepcopy(physical)
    probe.x = position.detach()
    current_sides = directional_side_indices_from_box(probe, quantile=0.1)
    return not any(
        not np.array_equal(np.sort(current_sides[name]), np.sort(indices))
        for name, indices in sides.items()
    )


def blocking_nodes(reference, physical, log_stiffness, position_q) -> torch.Tensor:
    _, position, _, lengths, box = model_geometry(reference, physical, log_stiffness, position_q)
    edge_index = reference.edge_index.long()
    original_lengths = reference.reference_context_edge_attr[:, 2]
    blocked = torch.zeros(len(position), dtype=torch.bool)
    invalid_edges = (lengths < 0.25 * original_lengths) | (lengths > 1.75 * original_lengths)
    blocked[edge_index[:, invalid_edges].flatten()] = True
    pairs = torch.triu_indices(len(position), len(position), 1)
    distances = position[pairs[1]] - position[pairs[0]]
    distances -= torch.round(distances / box) * box
    original = reference.reference_context_positions[:, :2]
    original_distances = original[pairs[1]] - original[pairs[0]]
    original_distances -= torch.round(original_distances / box) * box
    invalid_pairs = distances.norm(dim=1) < 0.1 * original_distances.norm(dim=1).min()
    blocked[pairs[:, invalid_pairs].flatten()] = True
    return blocked
