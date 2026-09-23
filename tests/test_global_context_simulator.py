import torch

from lss.dynamics.cold_graphs import rebuild_edge_features
from lss.dynamics.global_context_simulator import LocalGlobalLocalSimulator


def test_rebuild_edge_features_uses_minimum_image():
    positions = torch.tensor([[-0.49, 0.0, 0.0], [0.49, 0.0, 0.0]])
    edge_index = torch.tensor([[0], [1]])
    features = rebuild_edge_features(
        positions, edge_index, torch.ones(3), torch.tensor([30.0])
    )
    torch.testing.assert_close(features[0, :3], torch.tensor([0.02, 0.0, 0.0]))
    torch.testing.assert_close(features[0, 3], torch.tensor(0.02))
    torch.testing.assert_close(features[0, 4], torch.tensor(30.0))


def test_local_global_local_shapes_and_gradients():
    model = LocalGlobalLocalSimulator(
        node_feature_dim=9,
        edge_feature_dim=5,
        atom_type_count=3,
        hidden_dim=32,
        attention_layers=1,
        attention_heads=4,
    )
    nodes = torch.randn(6, 9)
    atom_types = torch.tensor([0, 1, 1, 2, 2, 1])
    edges = torch.randn(5, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    node_output, box_output = model(nodes, atom_types, edges, edge_index)
    assert node_output.shape == (6, 3)
    assert box_output.shape == (3,)
    (node_output.square().mean() + box_output.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_local_baseline_has_no_global_processor():
    model = LocalGlobalLocalSimulator(
        node_feature_dim=9,
        edge_feature_dim=5,
        atom_type_count=2,
        hidden_dim=32,
        attention_layers=0,
        attention_heads=4,
    )
    assert model.global_processor is None
