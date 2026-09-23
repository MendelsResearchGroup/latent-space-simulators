#!/usr/bin/env python3
"""Create the immutable code snapshot for the LJ-only edge study."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent
CODE = ROOT / "notebooks/results/ljonly_latent_influence_edges/code_v1"
FILES = ("protocol.json", "prepare_latent.py", "latent.pbs", "run_downstream.py", "downstream.pbs", "freeze.py")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if CODE.exists(): raise FileExistsError(CODE)
    CODE.mkdir(parents=True)
    for name in FILES: shutil.copy2(SOURCE / name, CODE / name)
    shutil.copy2(ROOT / "src/lss/past_experiments/dynamics/trajectory_propagator.py",
                 CODE / "trajectory_propagator.py")
    (CODE / "src/lss/dynamics").mkdir(parents=True)
    shutil.copy2(ROOT / "src/lss/dynamics/latent_influence_gnn.py",
                 CODE / "src/lss/dynamics/latent_influence_gnn.py")
    manifest = {str(path.relative_to(CODE)): sha(path) for path in sorted(CODE.rglob("*")) if path.is_file()}
    (CODE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(CODE)


if __name__ == "__main__": main()
