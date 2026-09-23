#!/usr/bin/env python3
"""Export matched latent-influence edges for one held-out LJ graph."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from run_downstream import OUT, canonical_raw, encode_records, load_scope, pilot


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
SEED = 123
OUTPUT = OUT / "downstream/evaluation/edge_example.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    _, validation, _ = canonical_raw(SEED)
    simulation = validation[0]
    reference = simulation[0]
    base = pilot.canonical_undirected(reference.edge_index.long())
    stiffness = reference.edge_attr[:, 3].float()
    springs = pilot.canonical_undirected(reference.edge_index[:, stiffness > 0].long())
    half = reference.reference_box_half_extent.float()
    center = reference.reference_box_center.float()
    output = {
        "seed": SEED,
        "source": "lj_noisy",
        "split": "validation",
        "original_index": int(reference.source_sim_id),
        "node_xy_physical": (reference.x[:, :2].float() * half + center).cpu(),
        "spring_edges": springs.cpu(),
        "gnn_base_edges": base.cpu(),
        "scope_edges": {},
    }
    for scope in ("ljonly60", "shared3x20"):
        ae_result, latent_cache, frozen, _, _ = load_scope(scope, SEED)
        row = encode_records([simulation], "validation", ae_result, latent_cache, smoke=False)[0]
        data = {"normalizers": ae_result["normalizers"],
                "latent_stats": latent_cache["latent_stats"]}
        scores = pilot.latent_score(ae_result["ae"], frozen["model"], row, data)
        extra = pilot.select_topk_nonphysical_edges(scores, base, 5)
        output["scope_edges"][scope] = {
            "added_edges": extra.cpu(),
            "scores": scores[extra[0], extra[1]].cpu(),
            "ae_sha256": sha(OUT / f"{scope}_s{SEED}/full/ae.pt"),
            "latent_model_sha256": sha(OUT / "latent" / f"{scope}_s{SEED}/full/latent_model.pt"),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, OUTPUT)
    metadata = {"artifact": str(OUTPUT.relative_to(ROOT)), "sha256": sha(OUTPUT),
                "network_id": output["original_index"], "seed": SEED,
                "spring_edges": int(springs.shape[1]), "gnn_base_edges": int(base.shape[1]),
                "added_edges": {scope: int(values["added_edges"].shape[1])
                                for scope, values in output["scope_edges"].items()}}
    (OUTPUT.parent / "edge_example.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
