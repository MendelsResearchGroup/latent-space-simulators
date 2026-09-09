"""A compact side-by-side animation of retained physical verification frames."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image

from .evaluation import sha256


def _segments(position, box, edge_index):
    length = box[:, 1] - box[:, 0]
    wrapped = box[:, 0] + np.mod(position - box[:, 0], length)
    start = wrapped[edge_index[0]]
    delta = wrapped[edge_index[1]] - start
    delta -= np.round(delta / length) * length
    pieces = [np.stack([start + np.array([x, y]) * length, start + delta + np.array([x, y]) * length], axis=1)
              for x in (-1, 0, 1) for y in (-1, 0, 1)]
    return wrapped, np.concatenate(pieces)


def render_comparison(output: Path, *, filename: str = "engineering_comparison.gif") -> Path:
    """Render the declared original/optimized trajectories in a four-second GIF."""
    output = Path(output)
    recipe = json.loads((output / "frozen_recipe.json").read_text())
    results = json.loads((output / "verification_results.json").read_text())
    names = ("original", "optimized")
    tracks = [np.load(results[name]["trajectory_path"]) for name in names]
    frame_count = int(results["original"]["frame_count"])
    if any(int(results[name]["frame_count"]) != frame_count for name in names):
        raise RuntimeError("Comparison requires matched physical frame counts")
    graph = torch.load(results["original"]["graph_path"], weights_only=False, map_location="cpu")
    edge_index = np.unique(np.sort(graph.edge_index.numpy(), axis=0), axis=1)
    boxes = [track["boxes"][:, :2, :2] for track in tracks]
    lower = np.min([box[:, :, 0].min(0) for box in boxes], axis=0)
    upper = np.max([box[:, :, 1].max(0) for box in boxes], axis=0)
    padding = (upper - lower) * 0.04
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.2), dpi=80, facecolor="#f7f4ee")
    canvas = FigureCanvasAgg(figure)
    figure.subplots_adjust(left=.035, right=.985, bottom=.085, top=.82, wspace=.08)
    figure.suptitle(f"{recipe['source']} {recipe['index']} final verification", fontsize=15, color="#38332d", y=.98)
    progress = figure.text(.5, .025, "Compression 0%", ha="center", fontsize=10, color="#6e655b")
    artists = []
    for axis, bounds, label, color, name in zip(axes, boxes, ("Original", "Adam optimized"), ("#8a8175", "#146b66"), names):
        axis.set_facecolor("#f7f4ee"); axis.set_aspect("equal")
        axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0]); axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
        axis.set_xticks([]); axis.set_yticks([])
        for spine in axis.spines.values(): spine.set_visible(False)
        axis.set_title(f"{label}\np-ratio {results[name]['physical_p_ratio']:+.3f}", fontsize=12, color=color, pad=8)
        rectangle = Rectangle(bounds[0][:, 0], *(bounds[0][:, 1] - bounds[0][:, 0]), fill=False, edgecolor=color, lw=.8)
        axis.add_patch(rectangle)
        lines = LineCollection([], colors=color, linewidths=.45, alpha=.60); axis.add_collection(lines)
        points = axis.scatter([], [], s=5, color=color, linewidths=0)
        artists.append((rectangle, lines, points))
    images = []
    for frame in range(frame_count):
        for track, bounds, (rectangle, lines, points) in zip(tracks, boxes, artists):
            box = bounds[frame]; position, segments = _segments(track["positions"][frame], box, edge_index)
            rectangle.set_bounds(*box[:, 0], *(box[:, 1] - box[:, 0])); lines.set_segments(segments); points.set_offsets(position)
        progress.set_text(f"Compression {100 * frame / (frame_count - 1):.0f}%")
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba()).copy()).convert("RGB"))
    plt.close(figure)
    palette = images[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
    quantized = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    target = output / filename
    durations = (np.diff(np.rint(np.linspace(0, 400, frame_count + 1))).astype(int) * 10).tolist()
    quantized[0].save(target, save_all=True, append_images=quantized[1:], duration=durations, loop=0, disposal=1, optimize=False)
    with Image.open(target) as image:
        total = 0
        for frame in range(image.n_frames): image.seek(frame); total += image.info["duration"]
        if image.n_frames != frame_count or total != 4000: raise RuntimeError("GIF frame or duration parity failed")
    metadata = {"path": str(target.resolve()), "sha256": sha256(target), "frames": frame_count, "duration_ms": 4000, "encoding": "64-color palette, no dithering, disposal 1"}
    (output / "animation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return target
