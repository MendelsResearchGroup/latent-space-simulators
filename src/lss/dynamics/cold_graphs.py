"""Direct loader and geometry helpers for Sergey's cold-rubber graphs."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


COLD_GRAPH_ROOT = Path("/rg/mendels_prj/s.sergey/graphs_cold")
REQUIRED_KEYS = {
    "x", "pos", "edge_index", "edge_attr", "time", "box", "box_tensor",
    "atom_ids", "atom_types", "molecule_ids", "bond_types",
}


class LegacyBox:
    """Unpickling target for the dataset's unavailable ``network_minimal.Box``."""


LegacyBox.__module__ = "network_minimal"


def _install_legacy_box_module() -> None:
    module = sys.modules.get("network_minimal")
    if module is None:
        module = types.ModuleType("network_minimal")
        sys.modules["network_minimal"] = module
    if not hasattr(module, "Box"):
        module.Box = LegacyBox


def clean_trajectory_ids(root: Path = COLD_GRAPH_ROOT) -> list[str]:
    """Return the declared complete 400-frame cohort without reordering it."""
    values = json.loads((root / "clean_400.json").read_text())
    if not isinstance(values, list) or not values or len(values) != len(set(values)):
        raise ValueError("clean_400.json must contain unique trajectory identifiers.")
    return [str(value) for value in values]


def load_cold_trajectory(identifier: str, root: Path = COLD_GRAPH_ROOT) -> list:
    """Load and validate one trajectory directly from its serialized PyG list."""
    _install_legacy_box_module()
    path = root / f"{identifier}.pt"
    frames = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(frames, list) or len(frames) != 400:
        raise ValueError(f"Expected 400 list frames in {path}.")
    first = frames[0]
    if set(first.keys()) != REQUIRED_KEYS:
        raise ValueError(f"Unexpected frame fields in {path}: {set(first.keys())}")
    nodes = int(first.pos.size(0))
    edges = int(first.edge_index.size(1))
    static = {
        "edge_index": first.edge_index,
        "atom_ids": first.atom_ids,
        "atom_types": first.atom_types,
        "molecule_ids": first.molecule_ids,
        "bond_types": first.bond_types,
    }
    previous_time: int | None = None
    for frame_index, frame in enumerate(frames):
        if set(frame.keys()) != REQUIRED_KEYS:
            raise ValueError(f"Frame {frame_index} fields changed in {path}.")
        if frame.pos.dtype != torch.float32 or tuple(frame.pos.shape) != (nodes, 3):
            raise ValueError(f"Invalid positions at frame {frame_index} in {path}.")
        if frame.edge_attr.dtype != torch.float32 or tuple(frame.edge_attr.shape) != (edges, 5):
            raise ValueError(f"Invalid edge attributes at frame {frame_index} in {path}.")
        if frame.edge_index.dtype != torch.int64 or tuple(frame.edge_index.shape) != (2, edges):
            raise ValueError(f"Invalid edge index at frame {frame_index} in {path}.")
        if frame.box_tensor.dtype != torch.float32 or tuple(frame.box_tensor.shape) != (3,):
            raise ValueError(f"Invalid box tensor at frame {frame_index} in {path}.")
        if not torch.equal(frame.x, frame.pos):
            raise ValueError(f"x and pos differ at frame {frame_index} in {path}.")
        for name, expected in static.items():
            if not torch.equal(getattr(frame, name), expected):
                raise ValueError(f"Static field {name} changed at frame {frame_index} in {path}.")
        current_time = int(frame.time)
        if previous_time is not None and current_time <= previous_time:
            raise ValueError(f"Times are not strictly increasing in {path}.")
        previous_time = current_time
    return frames


def rebuild_edge_features(
    positions: Tensor,
    edge_index: Tensor,
    box_lengths: Tensor,
    stiffness: Tensor,
) -> Tensor:
    """Rebuild the raw five-channel edge layout from a predicted state."""
    source, target = edge_index
    delta = positions[source] - positions[target]
    delta = delta - torch.round(delta / box_lengths) * box_lengths
    length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return torch.cat((delta, length, stiffness.reshape(-1, 1)), dim=-1)


def affine_strain(reference_positions: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Fit three independent affine coordinate slopes and return slope minus one."""
    reference = np.asarray(reference_positions, dtype=np.float64)
    current = np.asarray(positions, dtype=np.float64)
    if reference.shape != current.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Expected matching [nodes, 3] position arrays.")
    strain = []
    for component in range(3):
        design = np.column_stack((np.ones(len(reference)), reference[:, component]))
        strain.append(float(np.linalg.lstsq(design, current[:, component], rcond=None)[0][1] - 1.0))
    return np.asarray(strain)


def position_p_ratios(reference_positions: np.ndarray, positions: np.ndarray) -> tuple[float, float]:
    """Return post-fit transverse y/x and z/x ratios from positions only."""
    strain = affine_strain(reference_positions, positions)
    if not np.isfinite(strain).all() or abs(float(strain[0])) <= 1e-12:
        return float("nan"), float("nan")
    return float(-strain[1] / strain[0]), float(-strain[2] / strain[0])

