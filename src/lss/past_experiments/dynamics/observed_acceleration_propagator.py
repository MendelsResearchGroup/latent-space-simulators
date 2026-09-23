"""Archived observed-acceleration trajectory study model."""
import torch
from torch import nn
from .trajectory_propagator import ReferenceContext, mlp

FAMILIES = ('observed_acceleration',)


class ObservedAccelerationPropagator(nn.Module):
    def __init__(self, history_scale):
        super().__init__()
        self.observed_frames = 3
        self.context = ReferenceContext(18, 64, 'attention')
        self.acceleration = mlp(71, 64, 2)
        nn.init.zeros_(self.acceleration[-1].weight)
        nn.init.zeros_(self.acceleration[-1].bias)
        self.register_buffer('history_scale', history_scale.reshape(2).clone())

    def forward(self, observed_z, reference_tokens):
        assert observed_z.shape == (3, 2)
        context = self.context(reference_tokens)
        z = observed_z[2]
        velocity = (z-observed_z[0])/2
        states = list(observed_z.unbind())
        for step in range(2, 100):
            features = torch.cat((context, z, velocity/self.history_scale,
                                  observed_z[0], z.new_tensor([step/100])))
            acceleration = self.history_scale*self.acceleration(features)/100
            velocity = velocity+acceleration
            z = z+velocity
            states.append(z)
        return torch.stack(states)


def make_history_model(family, observed_frames, history_scale):
    assert family == 'observed_acceleration' and observed_frames == 3
    return ObservedAccelerationPropagator(history_scale)
