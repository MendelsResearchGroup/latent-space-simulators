"""Archived initial-graph trajectory-predictor study variants."""
import math

import torch
from torch import nn

FAMILIES = ('residual_mlp', 'inertial', 'gru', 'history_attention', 'direct_mlp', 'cubic')


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class ReferenceContext(nn.Module):
    """Pool reference-node embeddings and reference coordinates, without node order."""
    def __init__(self, token_dim, hidden, mode):
        super().__init__()
        self.mode = mode
        self.hidden = hidden
        if mode == 'mean':
            self.pool = mlp(token_dim, hidden, hidden)
        elif mode == 'attention':
            self.keys = nn.Linear(token_dim, hidden)
            self.values = nn.Linear(token_dim, hidden)
            self.queries = nn.Parameter(torch.randn(4, hidden) / math.sqrt(hidden))
            self.pool = mlp(4 * hidden, hidden, hidden)
        else:
            raise ValueError(mode)

    def forward(self, tokens):
        if self.mode == 'mean':
            return self.pool(tokens.mean(0))
        weights = (self.queries @ self.keys(tokens).T / math.sqrt(self.hidden)).softmax(-1)
        return self.pool((weights @ self.values(tokens)).flatten())


class TrajectoryPropagator(nn.Module):
    """Predict101 latent states from z0 and the reference graph alone.

    Recursive families integrate learned latent rates over a unit100-step horizon.
    Direct families evaluate a learned trajectory at prescribed normalized times.
    Time is a step index, not an observed deformation or response.
    """
    def __init__(self, family, context_mode, latent_dim=2, token_dim=18, hidden=64, horizon=100):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(family)
        self.family, self.horizon, self.latent_dim = family, horizon, latent_dim
        self.context = ReferenceContext(token_dim, hidden, context_mode)
        inputs = hidden + 2 * latent_dim + 1
        if family in ('residual_mlp', 'direct_mlp'):
            self.predict = mlp(inputs, hidden, latent_dim)
        elif family == 'cubic':
            self.predict = mlp(hidden + latent_dim, hidden, 3 * latent_dim)
        elif family == 'inertial':
            self.initial_velocity = mlp(hidden + latent_dim, hidden, latent_dim)
            self.predict = mlp(inputs + latent_dim, hidden, latent_dim)
        elif family == 'gru':
            self.initial_memory = mlp(hidden + latent_dim, hidden, hidden)
            self.memory = nn.GRUCell(inputs, hidden)
            self.predict = nn.Linear(hidden, latent_dim)
        elif family == 'history_attention':
            self.history_tokens = nn.Linear(latent_dim + 1, hidden)
            self.history_attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
            self.query = nn.Linear(inputs, hidden)
            self.predict = mlp(inputs + hidden, hidden, latent_dim)

    def forward(self, initial_z, reference_tokens):
        context = self.context(reference_tokens)
        times = torch.arange(self.horizon + 1, device=initial_z.device, dtype=initial_z.dtype) / self.horizon
        if self.family == 'cubic':
            coefficients = self.predict(torch.cat((context, initial_z))).reshape(3, self.latent_dim)
            basis = torch.stack((times, times.square(), times.pow(3)), -1)
            return initial_z + basis @ coefficients
        if self.family == 'direct_mlp':
            inputs = torch.cat((context.expand(self.horizon + 1, -1),
                                initial_z.expand(self.horizon + 1, -1),
                                torch.zeros(self.horizon + 1, self.latent_dim, device=initial_z.device, dtype=initial_z.dtype),
                                times[:, None]), -1)
            return initial_z + times[:, None] * self.predict(inputs)
        states = [initial_z]
        if self.family == 'inertial':
            velocity = self.initial_velocity(torch.cat((context, initial_z)))
        if self.family == 'gru':
            memory = self.initial_memory(torch.cat((context, initial_z)))
        for step in range(self.horizon):
            current = states[-1]
            inputs = torch.cat((context, initial_z, current - initial_z, times[step:step + 1]))
            if self.family == 'residual_mlp':
                rate = self.predict(inputs)
            elif self.family == 'inertial':
                velocity = velocity + self.predict(torch.cat((inputs, velocity))) / self.horizon
                rate = velocity
            elif self.family == 'gru':
                memory = self.memory(inputs, memory)
                rate = self.predict(memory)
            else:
                start = max(0, step - 7)
                tokens = self.history_tokens(torch.cat((torch.stack(states[start:]) - initial_z,
                                                       times[start:step + 1, None]), -1)).unsqueeze(0)
                query = self.query(inputs).reshape(1, 1, -1)
                attended, _ = self.history_attention(query, tokens, tokens, need_weights=False)
                rate = self.predict(torch.cat((inputs, attended.flatten())))
            states.append(current + rate / self.horizon)
        return torch.stack(states)


def make_trajectory_propagator(family, context_mode, latent_dim=2, token_dim=18, hidden=64, horizon=100):
    return TrajectoryPropagator(family, context_mode, latent_dim, token_dim, hidden, horizon)
