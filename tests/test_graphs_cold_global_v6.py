import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("cold_v6",ROOT/"hpc/experiments/graphs_cold_global_v6/run.py")
V6=importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None; SPEC.loader.exec_module(V6)

def test_deterministic_graph_batches_and_final_partial_batch():
    ids=[f"n{i}" for i in range(18)]
    first=V6.shuffled_batches(ids,np.random.default_rng(123),8)
    second=V6.shuffled_batches(ids,np.random.default_rng(123),8)
    assert first==second and [len(batch) for batch in first]==[8,8,2]
    assert sorted(identifier for batch in first for identifier in batch)==sorted(ids)

def test_gradient_scaling_uses_actual_batch_size():
    parameter=torch.tensor(2.0,requires_grad=True)
    losses=[parameter.square(),3*parameter.square()]
    sum(V6.scaled_graph_loss(loss,2) for loss in losses).backward()
    assert parameter.grad.item()==8.0
    parameter=torch.tensor(2.0,requires_grad=True)
    V6.scaled_graph_loss(3*parameter.square(),1).backward()
    assert parameter.grad.item()==12.0

def test_validation_boundary_positive_boxes_and_protocol():
    with pytest.raises(RuntimeError): V6.v5.assert_validation_boundary({"validation":["v"],"final_test":["f"]},["f"])
    assert torch.all(V6.v5.positive_box(torch.tensor([-1.0,0.0]),10.0)>0)
    protocol=json.loads((ROOT/"hpc/experiments/graphs_cold_global_v6/protocol.json").read_text())
    assert protocol["training"]["graph_trajectory_gradient_accumulation"]==8
    assert (protocol["training"]["max_epochs"],protocol["training"]["min_epochs"],protocol["training"]["patience"])==(200,50,20)
