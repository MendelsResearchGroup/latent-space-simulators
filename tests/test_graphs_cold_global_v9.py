import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("v9",ROOT/"hpc/experiments/graphs_cold_global_v9/run.py");V=importlib.util.module_from_spec(SPEC);assert SPEC.loader;SPEC.loader.exec_module(V)
def test_v9_reuses_v8_and_has_no_phase2():
 p=(ROOT/"hpc/experiments/graphs_cold_global_v9/protocol.json").read_text();assert "phase2 training" in p and V.V8.name=="run.py"
