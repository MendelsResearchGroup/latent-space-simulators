"""Unit checks for the reference-anchored v3 target definition."""
import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("cold_v3",ROOT/"hpc/experiments/graphs_cold_global_v3/run.py")
V3=importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V3)

def frame(pos, box):
    return {"pos":torch.tensor(pos,dtype=torch.float32),"box_tensor":torch.tensor(box,dtype=torch.float32)}

def test_reference_anchored_total_state_target_is_minimum_image_fractional():
    initial=frame([[9.0,2.0,3.0]],[10.0,10.0,10.0])
    following=frame([[1.0,5.0,8.0]],[10.0,12.0,14.0])
    node,box=V3.targets(initial,following)
    torch.testing.assert_close(node,torch.tensor([[0.2,0.21666667,0.27142859]]))
    torch.testing.assert_close(box,torch.tensor([2.0,4.0]))
