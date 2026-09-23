import torch

from lss.dynamics.fractional_contact_simulator import (
    FractionalContactSimulator,
    build_interaction_graph,
    fractional_coordinates,
    unit_minimum_image,
)


def test_fractional_coordinates_and_unit_minimum_image():
    positions = torch.tensor([[0.9, -1.8, 0.0]])
    box = torch.tensor([2.0, 4.0, 1.0])
    torch.testing.assert_close(fractional_coordinates(positions, box), torch.tensor([[0.45, -0.45, 0.0]]))
    torch.testing.assert_close(unit_minimum_image(torch.tensor([[0.9, -0.9, 0.1]])), torch.tensor([[-0.1, 0.1, 0.1]]))


def test_interaction_graph_adds_directed_contacts_without_duplicate_bonds():
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]])
    bonds = torch.tensor([[0], [1]])
    features, index = build_interaction_graph(
        positions,
        bonds,
        torch.tensor([4.0, 4.0, 4.0]),
        torch.tensor([30.0]),
        contact_cutoff=1.5,
    )
    pairs = set(map(tuple, index.t().tolist()))
    assert pairs == {(0, 1), (1, 0), (0, 2), (2, 0)}
    assert int(features[:, 4].sum()) == 2
    assert sorted(features[:, 5].tolist()) == [0.0, 0.0, 30.0, 30.0]


def test_fractional_contact_simulator_shapes_and_recurrent_state():
    model = FractionalContactSimulator(hidden_dim=32, attention_heads=4, attention_layers=1, recurrent=True)
    node_features = torch.randn(5, 9)
    atom_types = torch.tensor([0, 1, 0, 1, 0])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_features = torch.randn(4, 6)
    node_output, box_output, hidden = model(
        node_features,
        atom_types,
        edge_features,
        edge_index,
        torch.randn(9),
    )
    assert node_output.shape == (5, 3)
    assert box_output.shape == (2,)
    assert hidden.shape == (5, 32)
    (node_output.square().mean() + box_output.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
