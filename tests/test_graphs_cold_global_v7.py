import importlib.util
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("v7",ROOT/"hpc/experiments/graphs_cold_global_v7/run.py"); V=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(V)
def test_initial_motion_encoder_shapes():
 m=V.InitialMotionBoxResponse(hidden_dim=32,attention_heads=4); e=m.encode(torch.rand(4,6),torch.tensor([0,1,0,1]),torch.rand(5,6),torch.tensor([[0,1,2,3,0],[1,2,3,0,2]])); assert e.shape==(32,) and m.decode(e,torch.rand(3,7)).shape==(3,2)
def test_context_has_prescribed_x_and_observed_xyz_velocity_only():
 f0={"box_tensor":torch.tensor([10.,20.,30.]),"time":torch.tensor(0.)}; f1={"box_tensor":torch.tensor([12.,999.,-999.]),"time":torch.tensor(200.)}; value=V.contexts([f0,f1],[5])[0]
 assert value.shape==(7,) and value[3].item()==20.0
 torch.testing.assert_close(value[4:],torch.tensor([.01,4.895,-5.145]))
def test_final_guard_and_positive_box():
 with pytest.raises(RuntimeError): V.boundary({"validation":["v"],"final_test":["f"]},["f"])
 assert torch.all(torch.nn.functional.softplus(torch.tensor([-10.,0.]))>0)
