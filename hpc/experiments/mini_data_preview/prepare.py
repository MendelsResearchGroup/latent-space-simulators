#!/usr/bin/env python3
"""Prepare raw, paired 200-frame mini-network preview data and GIFs.

This is a data-only audit: it reads the serialized physical trajectories and
never constructs, trains, or evaluates an ML model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
OUT = ROOT / "notebooks/results/mini_data_preview"
GIF_DIR = ROOT / "notebooks/dynamics/Mini data — original and mixed temperature"
MANIFEST = ROOT / "notebooks/results/mini_shared/data_manifest.json"
THERMAL = ROOT / "hpc/experiments/ml_only_engineering"
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "MetaForge/src"), str(THERMAL)]
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from lss.engineering.animation import _segments
from thermal_metric import frame_time_ols_p_ratio

DATA = {
    "node": {
        "original": ROOT.parent / "data/mini_node_optimized_200_200.pt",
        "mixed_temp": ROOT.parent / "data/mini_node_optimized_mixed_temp_200_200.pt",
    },
    "stiffness": {
        "original": ROOT.parent / "data/mini_stiff_optimized_200_200.pt",
        "mixed_temp": ROOT.parent / "data/mini_stiff_optimized_mixed_temp_200_200.pt",
    },
}
CONTRACT_NETWORKS = ["source", "trajectory_id", "temperature_K", "p_ratio_endpoint", "p_ratio_fit", "fit_valid_frames", "fit_total_frames", "fitted_axial_total_strain", "fit_r2_x", "fit_r2_y"]
CONTRACT_SERIES = ["source", "trajectory_id", "frame", "strain_x", "strain_y", "p_ratio_endpoint", "p_ratio_fit"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, Path): return str(value)
    return value


def box_array(graph) -> np.ndarray:
    box = graph.box
    return np.asarray([[float(box.x1), float(box.x2)], [float(box.y1), float(box.y2)]], dtype=float)


def positions(sim) -> np.ndarray:
    return np.stack([frame.x[:, :2].detach().cpu().numpy().astype(float) for frame in sim])


def source_name(category: str, variant: str) -> str:
    return f"{category}_{'mixed' if variant == 'mixed_temp' else variant}"


def source_id(record: dict) -> int:
    return int(record["source_index"])


def selected_records(manifest: dict, category: str) -> list[dict]:
    records = manifest["sources"][category]["paired_original_mixed"]["records"]
    output = []
    for temperature in (0.1, 30.0):
        matched = [row for row in records if float(row["temperature_K"]) == temperature]
        if not matched: raise ValueError(f"{category}: no {temperature} K source")
        output.append(min(matched, key=lambda row: int(row["source_index"])))
    return output


def metrics(sim) -> tuple[dict, list[dict], dict]:
    if len(sim) != 200: raise ValueError(f"Expected 200 frames, got {len(sim)}")
    physical = positions(sim)  # Raw serialized physical coordinates; no inversion/normalization.
    sides = directional_side_indices_from_box(sim[0], quantile=.10)
    full = frame_time_ols_p_ratio(physical, sides, frame_index=np.arange(200), min_frames=8, min_axial_total_strain=1e-3)
    endpoint = np.asarray([float(calc_p_ratio_rollout_sides([sim[0], sim[frame]], -1, side_idx=sides)) for frame in range(200)])
    prefix = []
    for count in range(1, 201):
        fit = frame_time_ols_p_ratio(physical[:count], sides, frame_index=np.arange(count), min_frames=8, min_axial_total_strain=1e-3)
        prefix.append(float(fit["p_ratio"]))
    widths, heights = np.asarray(full["width"], dtype=float), np.asarray(full["height"], dtype=float)
    diagnostic = {
        "initial_width": widths[0], "final_width": widths[-1], "initial_height": heights[0], "final_height": heights[-1],
        "node_width_ratio": widths[-1] / widths[0], "node_height_ratio": heights[-1] / heights[0],
        "initial_box": box_array(sim[0]).tolist(), "final_box": box_array(sim[-1]).tolist(),
    }
    diagnostic["box_width_ratio"] = (diagnostic["final_box"][0][1] - diagnostic["final_box"][0][0]) / (diagnostic["initial_box"][0][1] - diagnostic["initial_box"][0][0])
    diagnostic["box_height_ratio"] = (diagnostic["final_box"][1][1] - diagnostic["final_box"][1][0]) / (diagnostic["initial_box"][1][1] - diagnostic["initial_box"][1][0])
    diagnostic["x_compression_ratio_abs_difference"] = abs(diagnostic["box_width_ratio"] - diagnostic["node_width_ratio"])
    return full, [{"frame": frame, "strain_x": float(full["strain_x"][frame]), "strain_y": float(full["strain_y"][frame]), "p_ratio_endpoint": endpoint[frame], "p_ratio_fit": prefix[frame]} for frame in range(200)], diagnostic


def edges(graph) -> np.ndarray:
    edge_index = graph.edge_index.detach().cpu().numpy()
    return np.unique(np.sort(edge_index, axis=0), axis=1)


def render_pair(original, mixed, *, category: str, record: dict, target: Path) -> dict:
    if len(original) != len(mixed) or len(original) != 200: raise ValueError("GIF requires matched 200-frame pair")
    tracks = [original, mixed]
    all_boxes = np.asarray([[box_array(frame) for frame in track] for track in tracks])
    lower = all_boxes[:, :, :, 0].min(axis=(0, 1)); upper = all_boxes[:, :, :, 1].max(axis=(0, 1))
    padding = (upper - lower) * .04
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.2), dpi=80, facecolor="#f7f4ee")
    canvas = FigureCanvasAgg(figure); figure.subplots_adjust(left=.035, right=.985, bottom=.085, top=.80, wspace=.08)
    temperature = float(record["temperature_K"])
    figure.suptitle(f"{category.title()} network {source_id(record)}: original and mixed {temperature:g} K", fontsize=14, color="#38332d", y=.98)
    progress = figure.text(.5, .025, "", ha="center", fontsize=10, color="#6e655b")
    artists = []
    for axis, track, label, color in zip(axes, tracks, ("Original", "Mixed temperature"), ("#8a8175", "#146b66")):
        axis.set_facecolor("#f7f4ee"); axis.set_aspect("equal"); axis.set_xlim(lower[0]-padding[0], upper[0]+padding[0]); axis.set_ylim(lower[1]-padding[1], upper[1]+padding[1]); axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_visible(False)
        axis.set_title(label, fontsize=12, color=color, pad=8)
        initial = box_array(track[0]); rectangle = Rectangle(initial[:, 0], *(initial[:, 1]-initial[:, 0]), fill=False, edgecolor=color, lw=.8); axis.add_patch(rectangle)
        lines = LineCollection([], colors=color, linewidths=.45, alpha=.60); axis.add_collection(lines); points = axis.scatter([], [], s=5, color=color, linewidths=0)
        artists.append((rectangle, lines, points, edges(track[0])))
    images = []
    for frame in range(200):
        for track, (rectangle, lines, points, edge_index) in zip(tracks, artists):
            box = box_array(track[frame]); xy = track[frame].x[:, :2].detach().cpu().numpy(); wrapped, segments = _segments(xy, box, edge_index)
            rectangle.set_bounds(*box[:, 0], *(box[:, 1]-box[:, 0])); lines.set_segments(segments); points.set_offsets(wrapped)
        progress.set_text(f"Frame {frame:03d}/199   paired mixed temperature {temperature:g} K")
        canvas.draw(); images.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB"))
    plt.close(figure)
    palette = images[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = (np.diff(np.rint(np.linspace(0, 300, 201))).astype(int) * 10).tolist()
    target.parent.mkdir(parents=True, exist_ok=True); quantized[0].save(target, save_all=True, append_images=quantized[1:], duration=durations, loop=0, disposal=1, optimize=False)
    with Image.open(target) as image:
        total = 0
        for frame in range(image.n_frames): image.seek(frame); total += image.info["duration"]
        if image.n_frames != 200 or total != 3000: raise RuntimeError(f"GIF parity failed: {image.n_frames} frames, {total} ms")
    return {"path": str(target.resolve()), "sha256": sha256(target), "frames": 200, "duration_ms": 3000, "durations_ms": {str(value): durations.count(value) for value in sorted(set(durations))}, "edges": "minimum-image wrapped segments tiled over neighbor boxes via lss.engineering.animation._segments"}


def run(mode: str) -> None:
    manifest = json.loads(MANIFEST.read_text())
    output = OUT / "pilot" if mode == "pilot" else OUT
    output.mkdir(parents=True, exist_ok=True)
    rows, series, diagnostics, examples, gifs = [], [], {}, [], {}
    categories = ("node",) if mode == "pilot" else ("node", "stiffness")
    for category in categories:
        chosen = selected_records(manifest, category)
        selected = chosen[:1] if mode == "pilot" else chosen
        original = torch.load(DATA[category]["original"], map_location="cpu", weights_only=False, mmap=True)
        mixed = torch.load(DATA[category]["mixed_temp"], map_location="cpu", weights_only=False, mmap=True)
        records = manifest["sources"][category]["paired_original_mixed"]["records"]
        selected_by_index = {int(record["source_index"]): record for record in selected}
        indices = sorted(selected_by_index) if mode == "pilot" else range(200)
        for index in indices:
            record = records[index]
            index, trajectory_id, temperature = int(record["source_index"]), source_id(record), float(record["temperature_K"])
            for variant, data in (("original", original), ("mixed_temp", mixed)):
                full, frame_rows, diagnostic = metrics(data[index])
                source = source_name(category, variant)
                rows.append({"source": source, "trajectory_id": trajectory_id, "temperature_K": temperature if variant == "mixed_temp" else float("nan"), "p_ratio_endpoint": float(calc_p_ratio_rollout_sides([data[index][0], data[index][-1]], -1, side_idx=directional_side_indices_from_box(data[index][0], quantile=.10))), "p_ratio_fit": full["p_ratio"], "fit_valid_frames": full["valid_frames"], "fit_total_frames": full["total_frames"], "fitted_axial_total_strain": full["fitted_axial_total_strain"], "fit_r2_x": full["fit_r2_x"], "fit_r2_y": full["fit_r2_y"]})
                series.extend({"source": source, "trajectory_id": trajectory_id, **frame} for frame in frame_rows)
                diagnostics[f"{source}:{trajectory_id}"] = diagnostic
            if index not in selected_by_index:
                continue
            name = f"{category}_{trajectory_id}_T{temperature:g}".replace(".", "p") + ".gif"
            gifs[name] = render_pair(original[index], mixed[index], category=category, record=record, target=GIF_DIR / name)
            examples.append({"category": category, "trajectory_id": trajectory_id, "temperature_K": temperature, "gif_path": gifs[name]["path"]})
        del original, mixed
    pd.DataFrame(rows, columns=CONTRACT_NETWORKS).to_csv(output / "networks.csv", index=False)
    pd.DataFrame(series, columns=CONTRACT_SERIES).to_csv(output / "frame_series.csv", index=False)
    metadata = {"status": "complete", "mode": mode, "pbs_job_id": os.environ.get("PBS_JOBID"), "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "coordinate_convention": "raw serialized physical graph.x[:, :2], used once; no position normalization or physical-coordinate inversion", "temperature_columns": "temperature_K is recorded only for regenerated mixed-temperature trajectories. Original trajectories have no recorded comparable temperature; their paired mixed temperature is retained in metadata examples and selection provenance.", "endpoint_estimator": "graph_utils.calc_p_ratio_rollout_sides with fixed frame-zero directional q=0.10 side indices", "fit_estimator": "thermal_metric.frame_time_ols_p_ratio; all 200 frames for networks.csv and prefix 1..200 fits for frame_series.csv; min_frames=8; min_axial_total_strain=1e-3", "selection": "before p-ratio computation: first source_index at each paired mixed temperature 0.1 and 30 K", "examples": examples, "gifs": gifs, "width_box_consistency": diagnostics, "simulation_provenance_caveat": "The mixed-temperature trajectories were regenerated by MetaForge from each source frame 0, with rebuilt rest lengths and the recorded zero-angle setup. Matched topology/source identity establishes pairing only; it does not establish an identical full mechanical protocol to the original trajectory.", "contracts": {"networks_csv": CONTRACT_NETWORKS, "frame_series_csv": CONTRACT_SERIES}, "inputs": {category: {variant: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for variant, path in variants.items()} for category, variants in DATA.items()}, "provenance": {"mini_shared_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)}, "data_audit": {"path": str(ROOT / "hpc/experiments/mini_shared/data_audit.py"), "sha256": sha256(ROOT / "hpc/experiments/mini_shared/data_audit.py")}, "thermal_metric": {"path": str(THERMAL / "thermal_metric.py"), "sha256": sha256(THERMAL / "thermal_metric.py")}, "animation_helper": {"path": str(ROOT / "src/lss/engineering/animation.py"), "sha256": sha256(ROOT / "src/lss/engineering/animation.py")}, "prepare": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))}}}
    (output / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": mode, "output": str(output), "rows": len(rows), "series_rows": len(series), "gifs": len(gifs)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    run(parser.parse_args().mode)
