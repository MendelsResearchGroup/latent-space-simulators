import importlib.util
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("cold_v5",ROOT/"hpc/experiments/graphs_cold_global_v5/run.py")
V5=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(V5)

def test_ols_uses_coordinate_context_and_never_p_or_strain():
    protocol=(ROOT/"hpc/experiments/graphs_cold_global_v5/protocol.json").read_text().lower()
    assert "p-ratio targets" in protocol and "strain targets" in protocol
    assert "p_ratio" not in V5.fit_ols_and_normalizers.__code__.co_names
    assert "strain" not in V5.fit_ols_and_normalizers.__code__.co_names

def test_zero_residual_recovers_constrained_ols_and_boxes_are_positive():
    initial=torch.tensor([5.0,7.0]); ols=torch.tensor([[1.0,-2.0],[3.0,4.0]]); zero=torch.zeros_like(ols)
    recovered=V5.combine_box(initial,ols,zero,10.0)
    control=V5.combine_box(initial,ols,torch.zeros_like(ols),10.0)
    torch.testing.assert_close(recovered,control)
    assert torch.all(recovered>0)
    assert torch.all(V5.positive_box(torch.tensor([-1.0,0.0]),10.0)>0)

def test_final_test_guard_rejects_final_ids():
    split={"train":["a"],"validation":["b"],"final_test":["c"]}
    V5.assert_validation_boundary(split,["b"])
    with pytest.raises(RuntimeError,match="final_test"):
        V5.assert_validation_boundary(split,["c"])
