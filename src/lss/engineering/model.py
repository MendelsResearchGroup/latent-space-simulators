"""Frozen learned-model and pristine-network loading for ML-only engineering."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import torch

from lss.data import (
    normalize_dataset_edges,
    normalize_trajectory_to_reference_box,
    tag_simulation_source,
)
from lss.dynamics.training import encode_frame_latent


@dataclass(frozen=True)
class FrozenModel:
    autoencoder: torch.nn.Module
    normalizers: dict[str, torch.Tensor]
    readout: dict[str, torch.Tensor]
    bundle_path: Path
    bundle_sha256: str


@dataclass(frozen=True)
class Network:
    reference: object
    physical: object
    index: int
    source: str


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(bundle_path: Path) -> FrozenModel:
    """Load and freeze the prepared AE and its pre-existing z1 affine readout."""
    bundle_path = Path(bundle_path)
    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    autoencoder = bundle["ae"]
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    return FrozenModel(
        autoencoder=autoencoder,
        normalizers=bundle["normalizers"],
        readout=bundle["policies"]["z1"],
        bundle_path=bundle_path.resolve(),
        bundle_sha256=_sha256(bundle_path),
    )


def load_network(dataset_path: Path, index: int, source: str) -> Network:
    """Load only raw frame zero and reproduce the shared in-memory preparation."""
    raw = torch.load(Path(dataset_path), weights_only=False, map_location="cpu", mmap=True)
    physical = deepcopy(raw[int(index)][0])
    trajectory = [deepcopy(raw[int(index)][0])]
    normalize_dataset_edges([trajectory], edge_multiplicity=1, edge_vector_dim=2)
    normalize_trajectory_to_reference_box(trajectory, pos_dim=2)
    tag_simulation_source(trajectory, source)
    return Network(
        reference=trajectory[0],
        physical=physical,
        index=int(index),
        source=str(source),
    )


def score(model: FrozenModel, reference, edited_graph) -> torch.Tensor:
    """Evaluate the frozen z1 affine readout from the edited frame-zero graph."""
    latent = encode_frame_latent(
        model.autoencoder,
        [edited_graph],
        0,
        pos_dim=2,
        node_feature_mode="normalized_delta",
        normalizers=model.normalizers,
        device="cpu",
    )
    readout = model.readout
    return readout["weights"][0] + (
        (latent - readout["mean"]) / readout["scale"]
    ) @ readout["weights"][1:]
