import importlib.util
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("v8",ROOT/"hpc/experiments/graphs_cold_global_v8/run.py");V=importlib.util.module_from_spec(SPEC);assert SPEC.loader;SPEC.loader.exec_module(V)
def test_node_only_shapes_and_no_box_decoder():
 m=V.NodeOnlyGlobalSimulator(hidden_dim=32,attention_heads=4);assert not hasattr(m,"box_decoder");assert m(torch.rand(4,9),torch.tensor([0,1,0,1]),torch.rand(5,6),torch.tensor([[0,1,2,3,0],[1,2,3,0,2]])).shape==(4,3)
def test_zero_output_is_constant_velocity():
 previous=torch.tensor([[1.,2.,3.]]);current=torch.tensor([[2.,4.,6.]])
 torch.testing.assert_close(V.integrate_constant_velocity(previous,current,torch.zeros_like(current),torch.ones(3)),torch.tensor([[3.,6.,9.]]))
def test_boundary_and_protocol_forbid_box_targets():
 with pytest.raises(RuntimeError):V.boundary({"validation":["v"],"final_test":["f"]},["f"])
 assert "box targets" in (ROOT/"hpc/experiments/graphs_cold_global_v8/protocol.json").read_text()
