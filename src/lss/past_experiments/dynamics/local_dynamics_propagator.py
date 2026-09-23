"""Archived local-error versus rollout-drift study models."""
import math
import torch
from torch import nn
from .trajectory_propagator import ReferenceContext, mlp

FAMILIES = ('state32', 'state64', 'state128', 'velocity64', 'velocity128', 'velocity_node64')


class LocalDynamicsPropagator(nn.Module):
    def __init__(self, family, increment_scale):
        super().__init__()
        self.observed_frames = 3
        self.use_velocity = family.startswith('velocity')
        self.node_conditioned = family == 'velocity_node64'
        hidden = {'state32':32, 'state64':64, 'state128':128,
                  'velocity64':64, 'velocity128':128, 'velocity_node64':64}[family]
        self.context = ReferenceContext(18, 64, 'attention')
        if self.node_conditioned:
            self.query = nn.Linear(7, 64)
        self.predict = mlp(71, hidden, 2)
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        self.register_buffer('increment_scale', increment_scale.reshape(2).clone())

    def predict_next(self, history, reference_tokens, steps):
        """history[B,3,2] ends at t; steps[B] contains t, never t+1 data."""
        assert history.ndim == 3 and history.shape[1:] == (3, 2)
        current = history[:, -1]
        velocity = (current-history[:,0])/2
        if not self.use_velocity:
            velocity = torch.zeros_like(velocity)
        state = torch.cat((current, velocity/self.increment_scale,
                           history[:,0], steps[:,None]/100), -1)
        # No-history models receive current state in both position slots.
        if not self.use_velocity:
            state = torch.cat((current, velocity, current, steps[:,None]/100), -1)
        if self.node_conditioned:
            weights = (self.query(state) @ self.context.keys(reference_tokens).T/math.sqrt(64)).softmax(-1)
            context = weights @ self.context.values(reference_tokens)
        else:
            context = self.context(reference_tokens).expand(len(history), -1)
        return current+self.increment_scale*self.predict(torch.cat((context,state),-1))

    def teacher_path(self, true_z, reference_tokens):
        histories = torch.stack([true_z[step-2:step+1] for step in range(2,100)])
        steps = torch.arange(2,100,device=true_z.device,dtype=true_z.dtype)
        return torch.cat((true_z[:3],self.predict_next(histories,reference_tokens,steps)))

    def forward(self, observed_z, reference_tokens):
        assert observed_z.shape == (3,2)
        states = list(observed_z.unbind())
        for step in range(2,100):
            prediction = self.predict_next(torch.stack(states[-3:])[None],reference_tokens,
                                           observed_z.new_tensor([step]))
            states.append(prediction[0])
        return torch.stack(states)


def make_history_model(family, observed_frames, history_scale):
    assert family in FAMILIES and observed_frames == 3
    return LocalDynamicsPropagator(family, history_scale)
