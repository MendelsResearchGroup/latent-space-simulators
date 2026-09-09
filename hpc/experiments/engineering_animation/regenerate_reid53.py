#!/usr/bin/env python3
"""Regenerate the verified Reid53 endpoint protocol with physical path dumps."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

DONOR = Path("/rg/mendels_prj/alexander.z/DL-course-project")
RESULT = Path(__file__).resolve().parents[3] / "notebooks/results/engineering_animation/reid53"
sys.path[:0] = [str(DONOR.parent / "MetaForge/src"), str(DONOR / "notebooks/network_design")]

from auxetic.utils import write_network
from auxetic.scripts import ElasticScript
from endpoint_physics import read_endpoint
from graph_utils import calc_p_ratio_rollout_sides
from graph_utils.box import Box


SCALE_X = 0.9700106999999999
STEPS = 120
EXPECTED = {"original": 0.09049821989813585, "engineered": -0.10344072336031537}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_dump(path, graph):
    """Read all LAMMPS custom dump frames as ordered unwrapped 2D positions."""
    lines = Path(path).read_text().splitlines()
    i = 0
    positions, boxes, graphs = [], [], []
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            raise ValueError(f"Expected timestep header at line {i + 1}")
        i += 2
        if lines[i] != "ITEM: NUMBER OF ATOMS":
            raise ValueError("Missing atom-count header")
        count = int(lines[i + 1]); i += 2
        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Unexpected box-bound header")
        if any(abs(float(lines[i + j].split()[2])) > 1e-14 for j in range(1, 4) if len(lines[i + j].split()) > 2):
            raise ValueError("Unexpected nonzero triclinic tilt in selected 2D protocol")
        bounds = [list(map(float, lines[i + j].split()[:2])) for j in range(1, 4)]
        i += 4
        names = lines[i].split()[2:]
        i += 1
        atoms = np.array([list(map(float, lines[i + j].split())) for j in range(count)])
        i += count
        atoms = atoms[np.argsort(atoms[:, names.index("id")])]
        if len(atoms) != len(graph.x):
            raise ValueError("Dump atom count differs from source graph")
        pos = atoms[:, [names.index("xu"), names.index("yu")]]
        out = deepcopy(graph)
        out.x = torch.tensor(pos, dtype=torch.float64)
        out.box = Box(*bounds[0], *bounds[1], *bounds[2])
        positions.append(pos); boxes.append(bounds[:2]); graphs.append(out)
    return np.asarray(positions), np.asarray(boxes), graphs


def simulate(variant):
    if variant not in EXPECTED:
        raise ValueError(variant)
    graph_path = RESULT / f"{variant}.pt"
    graph = torch.load(graph_path, weights_only=False, map_location="cpu")
    run = RESULT / "runs" / variant
    run.mkdir(parents=True, exist_ok=True)
    write_network(run, deepcopy(graph), mass=1e6, angles=0., box=graph.box)
    ElasticScript("network.lmp").write_to_file(str(run))
    step_scale = SCALE_X ** (1 / STEPS)
    script = f'''dimension 2
include init.mod
include potential.mod
fix planar all enforce2d
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1.0e-10 10000 100000
unfix relax
write_dump all custom frame_0.dump id xu yu modify sort id format float %.15g
variable increment loop {STEPS}
label compression
change_box all x scale {step_scale:.17g} remap units box
fix relax all box/relax y 0.0 vmax 0.001
minimize 0.0 1.0e-10 10000 100000
unfix relax
write_dump all custom frame_${{increment}}.dump id xu yu modify sort id format float %.15g
next increment
jump SELF compression
write_dump all custom final_endpoint.dump id x y xu yu modify sort id format float %.15g
'''
    (run / "in.animation").write_text(script)
    lammps = DONOR / ".runtime/lammps/bin/lmp"
    completed = subprocess.run([str(lammps), "-screen", "none", "-in", "in.animation"], cwd=run,
                               check=True, capture_output=True, text=True)
    (run / "stdout.txt").write_text(completed.stdout)
    parsed = [parse_dump(run / f"frame_{frame}.dump", graph) for frame in range(STEPS + 1)]
    positions = np.concatenate([item[0] for item in parsed])
    boxes = np.concatenate([item[1] for item in parsed])
    frames = [item[2][0] for item in parsed]
    if positions.shape[0] != STEPS + 1:
        raise ValueError(f"Expected {STEPS + 1} frames, found {positions.shape[0]}")
    pratio = np.full(STEPS + 1, np.nan)
    for frame in range(1, STEPS + 1):
        pratio[frame] = float(calc_p_ratio_rollout_sides([frames[0], frames[frame]], -1))
    endpoint = read_endpoint(run / "final_endpoint.dump", graph)
    endpoint_pratio = float(calc_p_ratio_rollout_sides([frames[0], endpoint], -1))
    if not np.isclose(pratio[-1], endpoint_pratio, atol=1e-12, rtol=0):
        raise ValueError("Dump-path endpoint disagrees with independently read endpoint")
    parity_error = abs(endpoint_pratio - EXPECTED[variant])
    if parity_error > 1e-6:
        raise ValueError(f"{variant} endpoint parity error {parity_error}")
    return positions, boxes, pratio, dict(
        expected_p_ratio=EXPECTED[variant], observed_p_ratio=endpoint_pratio,
        parity_error=parity_error, frame_count=int(positions.shape[0]),
        graph_sha256=sha256(graph_path), run_dir=str(run), lammps_binary=str(lammps),
        lammps_sha256=sha256(lammps), pbs_job_id=os.environ.get("PBS_JOBID"),
        runner_sha256=sha256(Path(__file__)),
        donor_endpoint_physics_sha256=sha256(DONOR / "notebooks/network_design/endpoint_physics.py"),
        donor_auxetic_utils_sha256=sha256(DONOR.parent / "MetaForge/src/auxetic/utils.py"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["original", "engineered"], required=True)
    args = parser.parse_args()
    positions, boxes, pratio, result = simulate(args.variant)
    np.savez_compressed(RESULT / f"{args.variant}_trajectory.npz", positions=positions, boxes=boxes,
                        pratio=pratio)
    (RESULT / f"{args.variant}_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
