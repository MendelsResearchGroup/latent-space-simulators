#!/usr/bin/env python3
"""Run the frozen MP2 compact-LJ recipe with only latent dimension set to two.

The donor runner and its code snapshot remain the authoritative training
implementation.  This adapter changes only its declared variant and output
study, so the recipe records both the adapter hash and the donor snapshot hash.
"""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(os.environ["LSS_PROJECT_ROOT"]).resolve()
DONOR = ROOT / "notebooks/results/compact_lj_reconstruction/code_v2/scripts/run_compact_lj_reconstruction.py"


def main() -> None:
    source = DONOR.read_text()
    marker = '    "mp2_r16_d4": ("message_passing_reference16", 4, True),\n'
    replacement = marker + '    "mp2_r16_d2": ("message_passing_reference16", 2, True),\n'
    if marker not in source:
        raise RuntimeError("Frozen donor variant table no longer matches the recorded adapter patch.")
    source = source.replace(marker, replacement, 1)
    old_output = 'ROOT / "notebooks/results/compact_lj_reconstruction" / name'
    if old_output not in source:
        raise RuntimeError("Frozen donor output expression no longer matches the recorded adapter patch.")
    source = source.replace(old_output, 'ROOT / "notebooks/results/compact_lj_mp2d" / name', 1)
    globals_dict = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), globals_dict)


if __name__ == "__main__":
    main()
