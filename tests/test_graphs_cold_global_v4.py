import importlib.util
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/lss/dynamics"))
from static_response_head import StaticGlobalBoxResponse
SPEC=importlib.util.spec_from_file_location("cold_v4",ROOT/"hpc/experiments/graphs_cold_global_v4/run.py")
V4=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(V4)

def test_pooling_shapes_and_decoder_batch():
    for pooling in ("mean","token"):
        model=StaticGlobalBoxResponse(pooling=pooling,hidden_dim=32,attention_heads=4)
        output,embedding=model(torch.rand(4,3),torch.tensor([0,1,0,1]),torch.rand(6,6),torch.tensor([[0,1,2,3,0,1],[1,2,3,0,2,3]]),torch.rand(5,5))
        assert embedding.shape==(32,) and output.shape==(5,2)

def test_context_and_target_exclude_yz_history_and_node_motion():
    initial={"pos":torch.tensor([[1.,2.,3.]]),"box_tensor":torch.tensor([10.,10.,10.]),"time":torch.tensor(0.)}
    one={"box_tensor":torch.tensor([11.,999.,-999.]),"time":torch.tensor(200.)}
    two={"box_tensor":torch.tensor([12.,-500.,500.]),"time":torch.tensor(400.)}
    changed_one={"box_tensor":torch.tensor([11.,1.,2.]),"time":torch.tensor(200.)}
    changed_two={"box_tensor":torch.tensor([12.,3.,4.]),"time":torch.tensor(400.)}
    torch.testing.assert_close(V4.context(initial,one,two,10),V4.context(initial,changed_one,changed_two,10))
    target_frame={"box_tensor":torch.tensor([20.,13.,14.])}
    torch.testing.assert_close(V4.target(initial,target_frame),torch.tensor([3.,4.]))

def test_affine_evaluation_uses_reference_fractional_positions():
    initial={"pos":torch.tensor([[2.,5.,8.]]),"box_tensor":torch.tensor([10.,10.,10.]),"time":torch.tensor(0.)}
    one={"box_tensor":torch.tensor([11.,7.,7.]),"time":torch.tensor(200.)}
    two={"box_tensor":torch.tensor([12.,8.,8.]),"time":torch.tensor(400.)}
    positions,box=V4.affine_prediction(initial,one,two,3,torch.tensor([11.,12.]))
    torch.testing.assert_close(box,torch.tensor([13.,11.,12.]))
    torch.testing.assert_close(positions,torch.tensor([[2.6,5.5,9.6]]))

def test_positive_box_parameterization_is_identity_in_physical_range():
    initial={"box_tensor":torch.tensor([10.,10.,10.])}
    normalizers={"target_mean":torch.tensor([0.,0.]),"target_std":torch.tensor([1.,1.])}
    physical=V4.positive_yz_box(initial,torch.tensor([[1.,2.],[-20.,-20.]]),normalizers)
    torch.testing.assert_close(physical[0],torch.tensor([11.000001,12.000001]),atol=2e-6,rtol=0)
    assert torch.all(physical[1]>0)
