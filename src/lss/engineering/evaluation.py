"""Freeze and physically verify ML-selected network designs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from auxetic.scripts import ElasticScript
from auxetic.utils import write_network
from graph_utils import calc_p_ratio_rollout_sides
from graph_utils.box import Box

from .optimization import OptimizationResult

COMPRESSION_SCALE = 0.9700106999999999
COMPRESSION_STEPS = 120


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    paths = [root / name for name in ("model.py", "geometry.py", "optimization.py", "evaluation.py", "animation.py")]
    paths.extend([root.parent / "data.py", root.parent / "graph.py"])
    paths.extend(sorted((root.parent / "dynamics").glob("*.py")))
    return {str(path.resolve()): sha256(path) for path in paths}


def freeze_final_designs(result: OptimizationResult, output: Path, *, dataset_path: Path, known_case_refactor_verification: bool) -> dict:
    """Write and hash the predeclared original and ML-selected final graphs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    graphs = {"original": result.original, "optimized": result.optimized}
    records = []
    for name, graph in graphs.items():
        path = output / f"{name}.pt"
        torch.save(graph, path)
        records.append({"name": name, "graph_path": str(path.resolve()), "graph_sha256": sha256(path)})
    history_path = output / "history.csv"
    result.history.to_csv(history_path, index=False)
    edits_path = output / "edits.pt"
    torch.save({"log_stiffness": result.log_stiffness, "position_q": result.position_q}, edits_path)
    recipe = {
        "known_case_refactor_verification": known_case_refactor_verification,
        "source": result.network.source,
        "index": result.network.index,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": sha256(Path(dataset_path)),
        "bundle_path": str(result.model.bundle_path),
        "bundle_sha256": result.model.bundle_sha256,
        "optimization": result.config.__dict__,
        "initial_score": result.initial_score,
        "best_score": result.best_score,
        "steps": result.steps,
        "stop_reason": result.stop_reason,
        "selection": "strict lowest frozen z1 score among initial and all feasible post-update iterates",
        "verification": {
            "protocol": "Reid athermal x compression with y box relaxation",
            "scale_x": COMPRESSION_SCALE,
            "increments": COMPRESSION_STEPS,
            "feedback_to_optimizer": False,
        },
        "designs": records,
        "history_path": str(history_path.resolve()),
        "history_sha256": sha256(history_path),
        "edits_path": str(edits_path.resolve()),
        "edits_sha256": sha256(edits_path),
        "source_hashes": _source_hashes(),
    }
    recipe_path = output / "frozen_recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
    manifest = {"recipe_path": str(recipe_path.resolve()), "recipe_sha256": sha256(recipe_path), "designs": records}
    (output / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _parse_dump(path: Path, graph):
    lines = Path(path).read_text().splitlines()
    cursor, positions, boxes, graphs = 0, [], [], []
    while cursor < len(lines):
        if lines[cursor] != "ITEM: TIMESTEP":
            raise ValueError("Invalid LAMMPS dump")
        count = int(lines[cursor + 3])
        cursor += 4
        bounds = [list(map(float, lines[cursor + offset].split()[:2])) for offset in range(1, 4)]
        cursor += 4
        names = lines[cursor].split()[2:]
        cursor += 1
        rows = np.array([list(map(float, lines[cursor + offset].split())) for offset in range(count)])
        cursor += count
        rows = rows[np.argsort(rows[:, names.index("id")])]
        position = rows[:, [names.index("xu"), names.index("yu")]]
        frame = deepcopy(graph)
        frame.x = torch.tensor(position, dtype=torch.float64)
        frame.box = Box(*bounds[0], *bounds[1], *bounds[2])
        positions.append(position)
        boxes.append(bounds[:2])
        graphs.append(frame)
    return np.asarray(positions), np.asarray(boxes), graphs


def _input_script() -> str:
    increment = COMPRESSION_SCALE ** (1 / COMPRESSION_STEPS)
    return f"""dimension 2
include init.mod
include potential.mod
fix planar all enforce2d
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1e-10 10000 100000
unfix relax
write_dump all custom frame_0.dump id xu yu modify sort id format float %.15g
variable increment loop {COMPRESSION_STEPS}
label compression
change_box all x scale {increment:.17g} remap units box
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1e-10 10000 100000
unfix relax
write_dump all custom frame_${{increment}}.dump id xu yu modify sort id format float %.15g
next increment
jump SELF compression
write_dump all custom final.dump id x y xu yu modify sort id format float %.15g
"""


def verify_frozen_designs(output: Path, lammps_path: Path) -> dict:
    """Run the fixed final-only LAMMPS protocol and retain all 121 frames."""
    output = Path(output)
    manifest = json.loads((output / "frozen_manifest.json").read_text())
    if sha256(Path(manifest["recipe_path"])) != manifest["recipe_sha256"]:
        raise RuntimeError("Frozen recipe hash mismatch")
    results = {}
    status_path = output / "verification_status.json"
    status = {"declared_designs": [record["name"] for record in manifest["designs"]], "completed": []}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    for record in manifest["designs"]:
        graph_path = Path(record["graph_path"])
        if sha256(graph_path) != record["graph_sha256"]:
            raise RuntimeError("Frozen graph hash mismatch")
        graph = torch.load(graph_path, weights_only=False, map_location="cpu")
        run_path = output / "verification" / record["name"]
        run_path.mkdir(parents=True, exist_ok=False)
        (run_path / "status.json").write_text(json.dumps({"status": "running", **record}, indent=2) + "\n")
        write_network(run_path, deepcopy(graph), mass=1e6, angles=0.0, box=graph.box)
        ElasticScript("network.lmp").write_to_file(str(run_path))
        (run_path / "in.engineering").write_text(_input_script())
        with (run_path / "stdout.txt").open("w") as stdout, (run_path / "stderr.txt").open("w") as stderr:
            try:
                subprocess.run([str(lammps_path), "-screen", "none", "-in", "in.engineering"], cwd=run_path, check=True, stdout=stdout, stderr=stderr, text=True)
            except subprocess.CalledProcessError as error:
                (run_path / "status.json").write_text(json.dumps({"status": "failed", "returncode": error.returncode, **record}, indent=2) + "\n")
                raise
        parsed = [_parse_dump(run_path / f"frame_{step}.dump", graph) for step in range(COMPRESSION_STEPS + 1)]
        positions = np.concatenate([item[0] for item in parsed])
        boxes = np.concatenate([item[1] for item in parsed])
        frames = [item[2][0] for item in parsed]
        p_ratio = np.array([np.nan] + [float(calc_p_ratio_rollout_sides([frames[0], frames[step]], -1)) for step in range(1, 121)])
        final_frame = _parse_dump(run_path / "final.dump", graph)[2][0]
        endpoint = float(calc_p_ratio_rollout_sides([frames[0], final_frame], -1))
        endpoint_parity_error = abs(float(p_ratio[-1]) - endpoint)
        if endpoint_parity_error > 1e-12:
            raise RuntimeError("Endpoint and retained-frame p-ratio disagree")
        np.savez_compressed(output / f"{record['name']}_trajectory.npz", positions=positions, boxes=boxes, p_ratio=p_ratio)
        results[record["name"]] = {
            **record,
            "trajectory_path": str((output / f"{record['name']}_trajectory.npz").resolve()),
            "trajectory_sha256": sha256(output / f"{record['name']}_trajectory.npz"),
            "physical_p_ratio": endpoint,
            "endpoint_parity_error": endpoint_parity_error,
            "frame_count": len(positions),
            "lammps_path": str(lammps_path),
            "lammps_sha256": sha256(lammps_path),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "verification_source_sha256": sha256(Path(__file__)),
        }
        (run_path / "status.json").write_text(json.dumps({"status": "completed", **results[record["name"]]}, indent=2) + "\n")
        status["completed"].append(record["name"])
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        (output / "verification_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
