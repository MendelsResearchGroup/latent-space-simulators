#!/usr/bin/env python3
"""Combine independently rerun Reid53 paths into the notebook payload."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "notebooks/results/engineering_animation/reid53"
DONOR = Path("/rg/mendels_prj/alexander.z/DL-course-project")
SOURCE = DONOR / "notebooks/results/network_design/geometry_target_v1/reid_53_stiffness"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = torch.load(OUT / "original.pt", weights_only=False, map_location="cpu")
    engineered = torch.load(OUT / "engineered.pt", weights_only=False, map_location="cpu")
    if not torch.equal(original.x, engineered.x):
        raise ValueError("The selected design must retain pristine initial geometry")
    if not torch.equal(original.edge_index, engineered.edge_index):
        raise ValueError("The selected design must retain pristine topology")
    k0 = original.edge_attr[:, 3].detach().cpu().numpy()
    k1 = engineered.edge_attr[:, 3].detach().cpu().numpy()
    if (k0 < 0).any() or (k1 < 0).any() or not np.array_equal(k0 == 0, k1 == 0):
        raise ValueError("Stiffness edits must preserve nonnegative springs and zero pattern")
    a = np.load(OUT / "original_trajectory.npz")
    b = np.load(OUT / "engineered_trajectory.npz")
    if a["positions"].shape != b["positions"].shape or a["positions"].shape[0] != 121:
        raise ValueError("Expected matched 121-frame physical paths")
    np.savez_compressed(
        OUT / "trajectories.npz",
        original_positions=a["positions"], engineered_positions=b["positions"],
        original_boxes=a["boxes"].reshape(-1, 4), engineered_boxes=b["boxes"].reshape(-1, 4),
        edge_index=original.edge_index.detach().cpu().numpy(),
        original_stiffness=k0, engineered_stiffness=k1,
        original_pratio=a["pratio"], engineered_pratio=b["pratio"],
    )
    payload_sha256 = sha256(OUT / "trajectories.npz")
    results = {v: json.loads((OUT / f"{v}_result.json").read_text()) for v in ("original", "engineered")}
    provenance = {
        "original": "pristine physical Reid trajectory 53 frame 0, saved as g.raw; not the continuation start graph",
        "continuation_start": "negative_reid53_v1/engineered.pt (SHA-256 2bf3d8e5755d79d5ae6c354411dae6c845030f2257259bf8768a8a31be46960f), used only to initialize geometry_target_v1 search",
        "final": "geometry_target_v1/reid_53_stiffness/engineered.pt, stiffness-only result; initial geometry and topology equal original",
    }
    metadata = {
        "source": "reid", "index": 53, "frame_count": 121,
        "compression_steps": 120, "scale_x": 0.9700106999999999,
        "physical_protocol": "2D quasistatic x compression; enforce2d; each increment change_box x scale 0.9700107**(1/120), y=0 box relaxation and LAMMPS minimization; frames written after initial relaxation and every minimization; p-ratio is first/each-frame unwrapped boundary-node ratio.",
        "endpoint_parity_tolerance": 1e-6, "endpoint_results": results,
        "valid_total": {"original": "120/120 post-initial frames", "engineered": "120/120 post-initial frames"},
        "finite_p_ratio_counts": {v: f"{int(np.isfinite(np.load(OUT / f'{v}_trajectory.npz')['pratio']).sum())}/120" for v in ("original", "engineered")},
        "trajectories_npz_sha256": payload_sha256,
        "provenance": provenance,
        "source_paths": {"design": str(SOURCE), "raw_data": str(DONOR / "data/reid_200_frames.pt")},
        "hashes": {name: sha256(OUT / name) for name in ("original.pt", "engineered.pt", "recipe.json", "summary.json", "confirmations.json", "final_edit.pt")},
        "checks": {"geometry_fixed": True, "topology_fixed": True, "zero_stiffness_pattern_fixed": True,
                   "all_stiffness_nonnegative": True},
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    provenance_manifest = {
        "design_source": str(SOURCE),
        "pristine_original": {"source": "data/reid_200_frames.pt trajectory 53 frame 0", "copy": "original.pt", "sha256": sha256(OUT / "original.pt")},
        "final_engineered": {"source": "geometry_target_v1/reid_53_stiffness/engineered.pt", "copy": "engineered.pt", "sha256": sha256(OUT / "engineered.pt")},
        "continuation_start_only": metadata["provenance"]["continuation_start"],
        "checkpoint_and_edit_provenance": {"frozen_ae_checkpoint_sha256": "62d9476c47f4024df0395d7dc14d4942a86f5c1baafb438ceb3ec6681be6dcc7", "edit": "final_edit.pt", "final_edit_sha256": sha256(OUT / "final_edit.pt")},
        "copied_donor_records": {name: sha256(OUT / name) for name in ("recipe.json", "summary.json", "confirmations.json")},
        "physical_reruns": {v: results[v] for v in ("original", "engineered")},
        "payload": {"path": "trajectories.npz", "sha256": payload_sha256},
    }
    (OUT / "provenance_manifest.json").write_text(json.dumps(provenance_manifest, indent=2))
    (OUT / "sha256sums.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sorted(OUT.glob("*")) if p.is_file()))


if __name__ == "__main__":
    main()
