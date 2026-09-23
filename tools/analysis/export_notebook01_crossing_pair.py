"""Export raw box motion for the two frozen notebook-01 near-collision networks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "notebooks/results/all4_coverage_1000/all4_one_each_retry_v5_validation30"
OUTPUT = ROOT / "notebooks/results/all4_coverage_1000/notebook01_crossing_pair"
PAIRS = (("node_low", 47), ("stiffness_low", 106))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_lengths(frame) -> tuple[float, float]:
    box = frame.box
    return float(box.x2 - box.x1), float(box.y2 - box.y1)


def main() -> None:
    recipe = json.loads((RESULTS / "recipe.json").read_text())
    sources = {item["name"]: item for item in recipe["source"]["dataset_mixture"]}
    rows = []
    provenance = {}
    for source, trajectory_id in PAIRS:
        path = Path(sources[source]["path"]).parent / "trajectories" / f"network_{trajectory_id:05d}.pt"
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        if len(trajectory) != 200:
            raise ValueError(f"Expected 200 frames in {path}")
        initial_lx, initial_ly = box_lengths(trajectory[0])
        provenance[source] = {"trajectory_id": trajectory_id, "path": str(path), "sha256": sha256(path)}
        for frame, graph in enumerate(trajectory):
            lx, ly = box_lengths(graph)
            delta_lx, delta_ly = lx - initial_lx, ly - initial_ly
            rows.append({
                "source": source,
                "trajectory_id": trajectory_id,
                "frame": frame,
                "md_step": int(graph.time),
                "Lx": lx,
                "Ly": ly,
                "delta_Lx": delta_lx,
                "delta_Ly": delta_ly,
                "strain_x": delta_lx / initial_lx,
                "strain_y": delta_ly / initial_ly,
                "box_p_ratio": float("nan") if frame == 0 else -delta_ly / delta_lx,
            })
    table = pd.DataFrame(rows)
    at_33 = table[table.frame.eq(33)].set_index("source")
    if not (at_33.loc["node_low", "delta_Ly"] > 0 > at_33.loc["stiffness_low", "delta_Ly"]):
        raise ValueError("Expected opposite frame-33 transverse box responses")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / "raw_box_motion.csv", index=False)
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "question": "Raw box response at a near-identical 2D latent point in notebook 01",
        "frozen_model_recipe": str(RESULTS / "recipe.json"),
        "frozen_model_recipe_sha256": sha256(RESULTS / "recipe.json"),
        "sources": provenance,
        "rows": len(table),
        "selected_frame": 33,
        "box_definition": "Lx=x2-x1, Ly=y2-y1; delta is relative to each frame-0 box",
    }, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT / 'raw_box_motion.csv'} ({len(table)} rows)")


if __name__ == "__main__":
    main()
