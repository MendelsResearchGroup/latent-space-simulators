#!/usr/bin/env python3
"""Execute one frozen runner from a JSON cell recipe."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

cell = json.loads(Path(os.environ["LSS_CELL_RECIPE"]).read_text())
code = Path(os.environ["LSS_CODE_ROOT"])
env = os.environ.copy()
env.update({"LSS_CODE_ROOT": str(code), "LSS_CODE_VERSION": cell["code_version"],
            "LSS_PROJECT_ROOT": cell["project_root"], "LSS_RESULTS_ROOT": cell["results_root"],
            "LSS_SPLIT_MANIFEST": cell["split_manifest"],
            "LSS_HISTORICAL_BASELINE_ROOT": cell["historical_results_root"],
            "LSS_HISTORICAL_AE": cell["historical_results_root"] + "/network_design/reid_lowT_2d_s786/ae.pt"})
raise SystemExit(subprocess.call([sys.executable, str(code / "runner" / cell["runner"]), *cell["arguments"]], env=env))
