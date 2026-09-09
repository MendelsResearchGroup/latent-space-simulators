"""Exact bounded joint Adam optimizer for a frozen learned engineering score."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
import torch

from .geometry import (
    blocking_nodes,
    boundary_constraints,
    local_length_scales,
    model_geometry,
    project_log_stiffness_,
    project_positions_,
    valid_geometry,
)
from .model import FrozenModel, Network, score


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 0.08
    stiffness_bound: float = 0.25
    patience: int = 50
    max_steps: int = 1200
    min_delta: float = 1e-5
    position_radius: float = 4.0
    mask_blocking_nodes: bool = True


@dataclass(frozen=True)
class OptimizationResult:
    original: object
    optimized: object
    history: pd.DataFrame
    log_stiffness: torch.Tensor
    position_q: torch.Tensor
    initial_score: float
    best_score: float
    steps: int
    stop_reason: str
    config: OptimizationConfig
    model: FrozenModel
    network: Network


def _export_physical(network: Network, log_stiffness: torch.Tensor, position_q: torch.Tensor):
    reference, original = network.reference, network.physical
    _, position, stiffness, _, box = model_geometry(reference, original, log_stiffness, position_q)
    physical = deepcopy(original)
    physical.x = position.detach().to(original.x)
    physical.pos = physical.x.clone()
    lookup = {tuple(sorted(pair)): edge for edge, pair in enumerate(reference.edge_index.T.tolist())}
    order = torch.tensor([lookup[tuple(sorted(pair))] for pair in original.edge_index.T.tolist()])
    vector = physical.x[physical.edge_index[1]] - physical.x[physical.edge_index[0]]
    vector -= torch.round(vector / box) * box
    physical.edge_attr = torch.cat(
        [vector, vector.norm(dim=1, keepdim=True), stiffness.detach()[order, None].to(vector)], dim=1
    )
    return physical


def optimize_network(model: FrozenModel, network: Network, config: OptimizationConfig) -> OptimizationResult:
    """Minimize frozen z1 using the sweep's raw-gradient joint Adam procedure."""
    reference, physical = network.reference, network.physical
    stiffness0 = reference.edge_attr[:, 3].detach()
    active = stiffness0 > 0
    sides, fixed, lower, upper = boundary_constraints(reference, physical)
    position0 = reference.reference_context_positions[:, :2].detach()
    scales = local_length_scales(reference).detach()
    movable = (~fixed)[:, None].expand_as(position0)
    log_stiffness = torch.zeros_like(stiffness0, requires_grad=True)
    position_q = torch.zeros_like(position0, requires_grad=True)
    if not valid_geometry(reference, physical, log_stiffness, position_q, sides):
        raise RuntimeError("Pristine geometry invalid")
    optimizer = torch.optim.Adam([log_stiffness, position_q], lr=config.learning_rate)
    initial_score = float(score(model, reference, model_geometry(reference, physical, log_stiffness, position_q)[0]).detach())
    best_score, reset_best = initial_score, initial_score
    best_u, best_q = log_stiffness.detach().clone(), position_q.detach().clone()
    history: list[dict] = []
    patience = 0
    stop_reason = "max_steps"

    for step in range(1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        current = score(model, reference, model_geometry(reference, physical, log_stiffness, position_q)[0])
        if not torch.isfinite(current):
            raise RuntimeError("Nonfinite current score")
        current.backward()
        if not torch.isfinite(log_stiffness.grad).all() or not torch.isfinite(position_q.grad).all():
            raise RuntimeError("Nonfinite Adam gradient")
        log_stiffness.grad[active] -= log_stiffness.grad[active].mean()
        log_stiffness.grad[~active] = 0
        position_q.grad[~movable] = 0
        last_u, last_q = log_stiffness.detach().clone(), position_q.detach().clone()
        optimizer.step()

        with torch.no_grad():
            proposal_u, proposal_q = log_stiffness.detach().clone(), position_q.detach().clone()
            project_log_stiffness_(proposal_u, active, config.stiffness_bound)
            project_positions_(proposal_q, position0, scales, fixed, lower, upper, config.position_radius)
            accepted, accepted_scale, halvings = False, 0.0, 16
            masked = torch.zeros(len(position_q), dtype=torch.bool)
            mask_passes = 0
            for halve in range(17):
                factor = 0.5**halve
                candidate_u = last_u + (proposal_u - last_u) * factor
                candidate_q = last_q + (proposal_q - last_q) * factor
                project_log_stiffness_(candidate_u, active, config.stiffness_bound)
                project_positions_(candidate_q, position0, scales, fixed, lower, upper, config.position_radius)
                if valid_geometry(reference, physical, candidate_u, candidate_q, sides):
                    log_stiffness.copy_(candidate_u)
                    position_q.copy_(candidate_q)
                    accepted, accepted_scale, halvings = True, factor, halve
                    break
            if not accepted and config.mask_blocking_nodes:
                for mask_passes in range(1, 17):
                    offenders = blocking_nodes(reference, physical, candidate_u, candidate_q)
                    new = offenders & ~masked & ~fixed
                    if not new.any():
                        break
                    masked |= new
                    proposal_q[masked] = last_q[masked]
                    project_positions_(proposal_q, position0, scales, fixed, lower, upper, config.position_radius)
                    for halve in range(17):
                        factor = 0.5**halve
                        candidate_u = last_u + (proposal_u - last_u) * factor
                        candidate_q = last_q + (proposal_q - last_q) * factor
                        project_log_stiffness_(candidate_u, active, config.stiffness_bound)
                        project_positions_(candidate_q, position0, scales, fixed, lower, upper, config.position_radius)
                        if valid_geometry(reference, physical, candidate_u, candidate_q, sides):
                            log_stiffness.copy_(candidate_u)
                            position_q.copy_(candidate_q)
                            accepted, accepted_scale, halvings = True, factor, halve
                            break
                    if accepted:
                        break
            if not accepted:
                log_stiffness.copy_(last_u)
                position_q.copy_(last_q)
            candidate = float(score(model, reference, model_geometry(reference, physical, log_stiffness, position_q)[0]))
        if not torch.isfinite(torch.tensor(candidate)):
            raise RuntimeError("Nonfinite post-update candidate score")
        strict_best = candidate < best_score
        if strict_best:
            best_score, best_u, best_q = candidate, log_stiffness.detach().clone(), position_q.detach().clone()
        material = candidate < reset_best - config.min_delta
        if material:
            reset_best, patience = candidate, 0
        else:
            patience += 1
        history.append(dict(step=step, predicted_p=candidate, best_score=best_score, patience_reset_best=reset_best,
                            patience=patience, accepted=accepted, accepted_scale=accepted_scale, halvings=halvings,
                            mask_passes=mask_passes, masked_nodes=int(masked.sum()), strict_best=strict_best,
                            material_improvement=material))
        if patience >= config.patience:
            stop_reason = f"patience_{config.patience}"
            break

    original = deepcopy(network.physical)
    optimized = _export_physical(network, best_u, best_q)
    return OptimizationResult(original=original, optimized=optimized, history=pd.DataFrame(history),
                              log_stiffness=best_u, position_q=best_q, initial_score=initial_score,
                              best_score=best_score, steps=len(history), stop_reason=stop_reason,
                              config=config, model=model, network=network)
