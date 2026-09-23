"""Boundary checks for teacher-input versus free-running diagnostics."""
import unittest
import torch
from lss.dynamics.local_dynamics_propagator import FAMILIES, make_history_model


class LocalDynamicsTests(unittest.TestCase):
    def test_input_boundaries_and_gradients(self):
        torch.manual_seed(123)
        z = torch.randn(101,2)*.1
        tokens = torch.randn(7,18)
        for family in FAMILIES:
            with self.subTest(family=family):
                model = make_history_model(family,3,torch.tensor([.02,.03]))
                teacher = model.teacher_path(z,tokens)
                torch.testing.assert_close(teacher[3:],z[2:100])
                rollout = model(z[:3],tokens)
                torch.testing.assert_close(rollout[3:],z[2].expand(98,2))
                assert torch.equal(rollout[:3],z[:3])
                with torch.no_grad():
                    model.predict[-1].weight.normal_(0,.02)
                teacher = model.teacher_path(z,tokens)
                changed = z.clone(); changed[30:]+=1
                # Prediction of frame30 uses only frames27/28/29.
                torch.testing.assert_close(teacher[:31],model.teacher_path(changed,tokens)[:31])
                torch.testing.assert_close(teacher,model.teacher_path(z,tokens.flip(0)))
                histories = z[:3][None].clone()
                before = model.predict_next(histories,tokens,z.new_tensor([2]))
                histories[:,0]+=1
                after = model.predict_next(histories,tokens,z.new_tensor([2]))
                if family.startswith('state'):
                    torch.testing.assert_close(before,after,atol=0,rtol=0)
                else:
                    self.assertGreater(float((before-after).abs().sum().detach()),0)
                teacher[3:].square().mean().backward()
                gradient = model.predict[-1].weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()),0)


if __name__ == '__main__':
    unittest.main()
