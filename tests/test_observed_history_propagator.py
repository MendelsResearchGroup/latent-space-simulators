"""Observed-prefix preservation and exclusion of unseen future frames."""
import unittest
import torch
from lss.dynamics.observed_history_propagator import make_history_model,linear_history_path


class HistoryTests(unittest.TestCase):
    def test_prefix_boundary_and_gradients(self):
        torch.manual_seed(123)
        truth=torch.randn(101,2)
        tokens=torch.randn(13,18)
        for family in ('history_mlp','velocity_residual'):
            for count in (1,3):
                with self.subTest(family=family,count=count):
                    model=make_history_model(family,count,torch.tensor([.1,.2]))
                    path=model(truth[:count],tokens)
                    torch.testing.assert_close(path[:count],truth[:count],rtol=0,atol=0)
                    altered=truth.clone();altered[count:]=torch.randn_like(altered[count:])*1000
                    torch.testing.assert_close(path,model(altered[:count],tokens),rtol=0,atol=0)
                    torch.testing.assert_close(path,model(truth[:count],tokens.flip(0)),rtol=1e-5,atol=1e-5)
                    path[3:].square().mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)
                    changed=truth[:count].clone();changed[-1]+=1
                    self.assertGreater(float((path[-1]-model(changed,tokens)[-1]).abs().sum()),0)

    def test_linear_motion_extrapolation(self):
        t=torch.arange(101).float()[:,None]
        truth=torch.tensor([[.4,-.6]])+t*torch.tensor([[.03,-.02]])
        torch.testing.assert_close(linear_history_path(truth[:3]),truth,rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
