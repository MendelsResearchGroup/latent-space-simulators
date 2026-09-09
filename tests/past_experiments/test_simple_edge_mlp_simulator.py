import torch

from lss.past_experiments.models.simple_edge_mlp_simulator import SimpleUndirectedEdgeMLPSimulator


def test_simple_edge_mlp_supports_variable_graph_sizes() -> None:
    model = SimpleUndirectedEdgeMLPSimulator(
        node_dim=6, edge_dim=13, hidden_size=16
    )
    for node_count, edge_index in (
        (3, torch.tensor([[0, 0], [1, 2]])),
        (5, torch.tensor([[0, 0, 1, 3], [1, 2, 4, 4]])),
    ):
        output = model(
            torch.randn(node_count, 6),
            torch.randn(edge_index.size(1), 13),
            edge_index,
        )
        assert output.shape == (node_count, 2)
        assert torch.isfinite(output).all()
