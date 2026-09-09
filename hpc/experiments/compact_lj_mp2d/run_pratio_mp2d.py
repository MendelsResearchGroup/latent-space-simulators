#!/usr/bin/env python3
"""Evaluate compact_lj_mp2d frozen checkpoints with the established metric."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
DONOR = ROOT / "hpc/experiments/compact_lj_pratio/run_pratio.py"


def main() -> None:
    source = DONOR.read_text()
    old = "assert study in {'compact_lj_reconstruction', 'compact_lj_spatial_decoder'}"
    new = "assert study in {'compact_lj_reconstruction', 'compact_lj_spatial_decoder', 'compact_lj_mp2d'}"
    if old not in source:
        raise RuntimeError("Established p-ratio evaluator study guard no longer matches adapter patch.")
    source = source.replace(old, new, 1)
    globals_dict = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), globals_dict)


if __name__ == "__main__":
    main()
