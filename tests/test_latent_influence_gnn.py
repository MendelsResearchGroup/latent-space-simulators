import torch

from lss.dynamics.latent_influence_gnn import (
    NormalizedAccelerationGNN,
    canonical_undirected,
    random_nonphysical_edges,
    select_topk_nonphysical_edges,
)


def test_topk_excludes_base_edges_and_random_control_matches_size():
    scores = torch.tensor([
        [0., 9., 8., 7., 6.],
        [9., 0., 5., 4., 3.],
        [8., 5., 0., 2., 1.],
        [7., 4., 2., 0., 8.],
        [6., 3., 1., 8., 0.],
    ])
    base = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    selected = select_topk_nonphysical_edges(scores, base, 1)
    base_set = {tuple(pair) for pair in canonical_undirected(base).t().tolist()}
    assert all(tuple(pair) not in base_set for pair in selected.t().tolist())
    random = random_nonphysical_edges(5, base, selected.shape[1], generator=torch.Generator().manual_seed(3))
    assert random.shape == selected.shape


def test_normalized_gnn_is_node_permutation_equivariant():
    torch.manual_seed(4)
    model = NormalizedAccelerationGNN(hidden_dim=16, steps=2)
    nodes = torch.randn(5, 8)
    edges = torch.randn(5, 10)
    index = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 4]])
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(5)
    expected = model(nodes, edges, index)
    actual = model(nodes[permutation], edges, inverse[index])
    torch.testing.assert_close(actual[inverse], expected, rtol=1e-5, atol=1e-6)


def test_model_contains_layer_normalization():
    model = NormalizedAccelerationGNN(hidden_dim=16, steps=2)
    assert sum(isinstance(module, torch.nn.LayerNorm) for module in model.modules()) == 6
