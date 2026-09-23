"""Archived observed-prefix trajectory study models."""
import torch
from torch import nn
from .trajectory_propagator import ReferenceContext, mlp

FAMILIES = ('history_mlp', 'velocity_residual')


class ObservedHistoryPropagator(nn.Module):
    def __init__(self, family, observed_frames, history_scale):
        super().__init__()
        assert family in FAMILIES and observed_frames in (1, 3)
        self.family, self.observed_frames = family, observed_frames
        self.context = ReferenceContext(18, 64, 'attention')
        self.predict = mlp(74, 64, 2)
        self.register_buffer('history_scale', history_scale.reshape(2).clone())

    def forward(self, observed_z, reference_tokens):
        assert observed_z.shape == (self.observed_frames, 2)
        initial, anchor = observed_z[0], observed_z[-1]
        if self.observed_frames == 3:
            delta1, delta2 = observed_z[1]-initial, anchor-observed_z[1]
            velocity = (anchor-initial)/2
        else:
            delta1 = delta2 = velocity = torch.zeros_like(initial)
        condition = torch.cat((self.context(reference_tokens), initial, anchor,
                               delta1/self.history_scale, delta2/self.history_scale))
        frames = torch.arange(101, device=initial.device, dtype=initial.dtype)
        elapsed = (frames-(self.observed_frames-1)).clamp_min(0)
        inputs = torch.cat((condition.expand(101, -1), elapsed[:,None]/100, frames[:,None]/100), -1)
        future = anchor+(elapsed[:,None]/100)*self.predict(inputs)
        if self.family == 'velocity_residual':
            future = future+elapsed[:,None]*velocity
        return torch.cat((observed_z, future[self.observed_frames:]), 0)


def make_history_model(family, observed_frames, history_scale):
    return ObservedHistoryPropagator(family, observed_frames, history_scale)


def linear_history_path(observed_z):
    """Constant latent velocity from the equally spaced frames0,1,2, anchored at2."""
    assert observed_z.shape == (3, 2)
    frames = torch.arange(3,101,device=observed_z.device,dtype=observed_z.dtype)
    future = observed_z[2]+(frames[:,None]-2)*(observed_z[2]-observed_z[0])/2
    return torch.cat((observed_z, future),0)
