"""Refresh the common source-wise AE/rollout response tables."""
import os
from pathlib import Path
import runpy

root = Path(os.environ.get('LSS_PROJECT_ROOT', Path(__file__).resolve().parents[3]))
runpy.run_path(str(root / 'hpc/experiments/compact_lj_response_rollout/collect.py'), run_name='__main__')
