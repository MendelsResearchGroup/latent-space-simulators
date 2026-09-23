"""Tiny, permutation-invariant edge MLP autoencoder for a bounded pilot.

No message passing or attention. The encoder's exact initial-state map is
mean_e,o [W2 tanh(W1 (([o*r_e, 0] - mean) / std)) + b2], where
o reverses only the two vector channels, not length or stiffness. Both edge
orientations are included so endpoint ordering has no effect. This is an
edge-distribution representation; it cannot distinguish all graph topologies.
"""

import torch
from torch import nn


class TinyEdgeAutoencoder(nn.Module):
    """Encoder 8->8->2 and node decoder 8->16->2: 268 parameters.

Inputs: physical reference edges [dx,dy,length,stiffness], evolving compact
edges [delta_dx,delta_dy,delta_length,relative_delta_length], reference node
positions, edge indices and node-to-graph indices. Output displacements are
in the runner's training-target standardized coordinates.
"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 2))
        self.register_buffer('feature_mean', torch.zeros(8))
        self.register_buffer('feature_std', torch.ones(8))
        self.register_buffer('position_mean', torch.zeros(2))
        self.register_buffer('position_std', torch.ones(2))
        self.register_buffer('reverse_sign', torch.tensor([-1., -1., 1., 1.]))

    def edge_inputs(self, ref_edge, delta_edge):
        if ref_edge.ndim != 2 or ref_edge.shape[1] != 4 or delta_edge.shape != ref_edge.shape:
            raise ValueError('Expected matching [edges, 4] reference/evolving features.')
        return torch.cat((torch.cat((ref_edge, delta_edge), dim=1),
                          torch.cat((ref_edge*self.reverse_sign,
                                     delta_edge*self.reverse_sign), dim=1)), dim=0)

    def set_normalization(self, feature_mean, feature_std, position_mean, position_std):
        for name, value, size in [('feature_mean', feature_mean, 8),
                                  ('feature_std', feature_std, 8),
                                  ('position_mean', position_mean, 2),
                                  ('position_std', position_std, 2)]:
            value = torch.as_tensor(value, device=self.feature_mean.device, dtype=self.feature_mean.dtype).reshape(-1)
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid {name}.')
            if name.endswith('std') and not (value > 0).all():
                raise ValueError(f'{name} must be positive (clamp constant channels when fitting).')
            getattr(self, name).copy_(value)

    def encode(self, ref_edge, delta_edge, edge_index, batch):
        edge_batch = batch[edge_index[0]]
        if not torch.equal(edge_batch, batch[edge_index[1]]):
            raise ValueError('An edge cannot connect different graphs in a batch.')
        edge_batch = torch.cat((edge_batch, edge_batch))
        inputs = (self.edge_inputs(ref_edge, delta_edge)-self.feature_mean)/self.feature_std
        edge_z = self.encoder(inputs)
        count = int(batch.max())+1
        z = edge_z.new_zeros((count, 2)).index_add(0, edge_batch, edge_z)
        weights = torch.bincount(edge_batch, minlength=count).to(z.dtype)
        if (weights == 0).any():
            raise ValueError('Every graph must have at least one edge.')
        return z/weights[:, None]

    def decode(self, z, ref_pos, ref_edge, edge_index, batch):
        # Static local context only: no evolving edge or target reaches decoder.
        directed_ref = torch.cat((ref_edge, ref_edge*self.reverse_sign), dim=0)
        directed_ref = (directed_ref-self.feature_mean[:4])/self.feature_std[:4]
        endpoints = torch.cat((edge_index[0], edge_index[1]))
        context = ref_pos.new_zeros((len(ref_pos), 4)).index_add(0, endpoints, directed_ref)
        degree = torch.bincount(endpoints, minlength=len(ref_pos)).to(ref_pos.dtype)
        context = context/degree.clamp_min(1)[:, None]
        position = (ref_pos-self.position_mean)/self.position_std
        return self.decoder(torch.cat((position, context, z[batch]), dim=1))

    def forward(self, ref_edge, delta_edge, ref_pos, edge_index, batch):
        z = self.encode(ref_edge, delta_edge, edge_index, batch)
        return self.decode(z, ref_pos, ref_edge, edge_index, batch), z
