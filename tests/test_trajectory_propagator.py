"""Input boundary, node-order invariance and differentiability of trajectory models."""
import unittest
import torch
from lss.dynamics.trajectory_propagator import FAMILIES, make_trajectory_propagator


class TrajectoryTests(unittest.TestCase):
    def test_all_families_preserve_initial_state_and_node_permutation(self):
        torch.manual_seed(123)
        initial = torch.tensor([0.4, -0.6])
        nodes = torch.randn(11, 18)
        permutation = torch.randperm(len(nodes))
        for family in FAMILIES:
            for context in ('mean', 'attention'):
                with self.subTest(family=family, context=context):
                    model = make_trajectory_propagator(family, context, hidden=16, horizon=10)
                    trajectory = model(initial, nodes)
                    self.assertEqual(trajectory.shape, (11, 2))
                    torch.testing.assert_close(trajectory[0], initial, rtol=0, atol=0)
                    torch.testing.assert_close(trajectory, model(initial, nodes[permutation]), atol=1e-6, rtol=1e-5)
                    trajectory[-1].square().sum().backward()
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
                    replacement = make_trajectory_propagator(family, context, hidden=16, horizon=10)
                    replacement.load_state_dict(model.state_dict())
                    torch.testing.assert_close(trajectory, replacement(initial, nodes), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
