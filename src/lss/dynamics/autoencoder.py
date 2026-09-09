"""Graph autoencoders for normalized displacement reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SimpleAttentionPool(nn.Module):
    def __init__(self, hidden_size: int, latent_tokens: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_tokens = int(latent_tokens)

        self.pool_queries = nn.Parameter(torch.randn(latent_tokens, hidden_size) * 0.01)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h: Tensor) -> Tensor:
        q = self.query_proj(self.pool_queries)  # [T, H]
        k = self.key_proj(h)  # [N, H]
        v = self.value_proj(h)  # [N, H]

        scores = (q @ k.transpose(0, 1)) / math.sqrt(self.hidden_size)  # [T, N]
        attn = torch.softmax(scores, dim=-1)  # [T, N]
        tokens = attn @ v  # [T, H]
        return tokens


class DirectLatentAttentionPool(nn.Module):
    """Use one learned query to produce each scalar latent coordinate."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.queries = nn.Parameter(torch.randn(latent_dim, hidden_size) * 0.01)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        q = self.query_proj(self.queries)  # [D, H]
        k = self.key_proj(tokens)  # [T, H]
        values = self.value_proj(tokens).transpose(0, 1)  # [D, T]
        attention = torch.softmax(
            (q @ k.transpose(0, 1)) / math.sqrt(self.hidden_size), dim=-1
        )  # [D, T]
        return (attention * values).sum(dim=-1)  # [D]


class PyramidAttentionPool(nn.Module):
    """Hierarchical attention reduction: N nodes -> 100 -> 20 -> latent_dim."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.nodes_to_100 = SimpleAttentionPool(hidden_size, 100)
        self.tokens_100_to_20 = SimpleAttentionPool(hidden_size, 20)
        self.tokens_20_to_latent = DirectLatentAttentionPool(
            hidden_size, latent_dim
        )

    def forward(self, node_features: Tensor) -> Tensor:
        tokens_100 = self.nodes_to_100(node_features)
        tokens_20 = self.tokens_100_to_20(tokens_100)
        return self.tokens_20_to_latent(tokens_20)


class SimpleMLPPool(nn.Module):
    """Pool a variable token set using MLP-produced weights, without Q/K attention."""

    def __init__(self, hidden_size: int, output_tokens: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_tokens),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        weights = torch.softmax(self.score_mlp(tokens).transpose(0, 1), dim=-1)
        return weights @ self.value_mlp(tokens)


class DirectLatentMLPPool(nn.Module):
    """Reduce tokens to scalar latent coordinates using MLP-produced weights."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )
        self.value_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        weights = torch.softmax(self.score_mlp(tokens).transpose(0, 1), dim=-1)
        values = self.value_proj(tokens).transpose(0, 1)
        return (weights * values).sum(dim=-1)


class PyramidMLPPool(nn.Module):
    """Hierarchical MLP-weighted reduction: N nodes -> 100 -> 20 -> latent_dim."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.nodes_to_100 = SimpleMLPPool(hidden_size, 100)
        self.tokens_100_to_20 = SimpleMLPPool(hidden_size, 20)
        self.tokens_20_to_latent = DirectLatentMLPPool(hidden_size, latent_dim)

    def forward(self, node_features: Tensor) -> Tensor:
        tokens_100 = self.nodes_to_100(node_features)
        tokens_20 = self.tokens_100_to_20(tokens_100)
        return self.tokens_20_to_latent(tokens_20)


class NodeDeltaAttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        pos_dim: int,
        edge_dim: int,
        hidden_size: int,
        latent_dim: int,
        latent_tokens: int,
        node_feature_dim: int | None = None,
        reconstruction_dim: int | None = None,
        message_passing_steps: int = 0,
        message_chunk_size: int = 8192,
        correct_edge_reversal: bool = False,
    ):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.latent_tokens = int(latent_tokens)
        self.node_feature_dim = int(node_feature_dim or pos_dim)
        self.reconstruction_dim = int(reconstruction_dim or pos_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.message_chunk_size = int(message_chunk_size)
        self.correct_edge_reversal = bool(correct_edge_reversal)
        if self.message_passing_steps < 0:
            raise ValueError("message_passing_steps must be non-negative.")
        if self.message_chunk_size < 1:
            raise ValueError("message_chunk_size must be positive.")

        self.edge_in = nn.Linear(edge_dim, hidden_size)
        self.ref_edge_in = nn.Linear(edge_dim, hidden_size)
        self.ref_node_in = nn.Linear(pos_dim + hidden_size, hidden_size)
        self.node_in = nn.Linear(self.node_feature_dim + hidden_size, hidden_size)
        self.message_edge_in = nn.Linear(edge_dim, hidden_size)
        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(self.message_passing_steps)
            ]
        )
        self.message_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(self.message_passing_steps)
            ]
        )
        self.message_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(self.message_passing_steps)]
        )
        # Filled from fitted edge normalizers by the experiment builder.  It
        # is recomputed from saved normalizers on load, so frozen legacy state
        # dictionaries remain loadable.
        self.register_buffer("edge_reverse_offset", torch.zeros(edge_dim), persistent=False)
        self.register_buffer("ref_edge_reverse_offset", torch.zeros(edge_dim), persistent=False)
        self.pool = PyramidAttentionPool(hidden_size, latent_dim)
        self.to_latent = None

        self.decoder_token_proj = nn.Linear(latent_dim, latent_tokens * hidden_size)
        self.decoder_query_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_key_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_value_proj = nn.Linear(hidden_size, hidden_size)
        self.node_decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.reconstruction_dim),
        )

    def encode_reference_graph(
        self, ref_pos: Tensor, ref_edge_attr: Tensor, edge_index: Tensor
    ) -> Tensor:
        edge_node = self.aggregate_edges(
            ref_edge_attr, edge_index, ref_pos.size(0), projection=self.ref_edge_in
        )
        return self.ref_node_in(torch.cat([ref_pos, edge_node], dim=-1))

    def aggregate_edges(
        self,
        edge_attr: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        *,
        projection: nn.Linear | None = None,
    ) -> Tensor:
        if edge_attr.numel() == 0:
            return torch.zeros(
                num_nodes, self.hidden_size, device=edge_attr.device, dtype=edge_attr.dtype
            )

        row, col = edge_index
        # Serialized datasets now carry one canonical edge per unordered pair.
        # Treat that edge as incident to both endpoints inside the model.  This
        # also remains compatible with older reciprocal edge lists: each
        # endpoint view is merely repeated and the mean is unchanged.
        reverse_attr = edge_attr.clone()
        if edge_attr.size(-1) == 4:
            directional_columns = [0, 1]
        elif edge_attr.size(-1) >= 11:
            # Expanded latent edge layout:
            # ref_vec, cur_vec, lengths/stretch/stiffness, raw_cur_vec, ...
            directional_columns = [0, 1, 2, 3, 9, 10]
        else:
            directional_columns = list(range(min(self.pos_dim, edge_attr.size(-1))))
        if self.correct_edge_reversal:
            reverse_offset = (
                self.ref_edge_reverse_offset
                if projection is self.ref_edge_in
                else self.edge_reverse_offset
            )
            reverse_attr[:, directional_columns] = (
                -edge_attr[:, directional_columns]
                + reverse_offset[directional_columns].to(edge_attr)
            )
        else:
            reverse_attr[:, directional_columns] *= -1
        endpoint = torch.cat([col, row])
        endpoint_attr = torch.cat([edge_attr, reverse_attr], dim=0)
        # Linear projection commutes with mean aggregation:
        # mean(W e + b) == W mean(e) + b. Pooling the compact raw features
        # first avoids a potentially enormous [N*(N-1), hidden_size]
        # intermediate for complete graphs without changing the result.
        node_sum = torch.zeros(
            num_nodes, edge_attr.size(-1), device=edge_attr.device, dtype=edge_attr.dtype
        )
        node_count = torch.zeros(
            num_nodes, 1, device=edge_attr.device, dtype=edge_attr.dtype
        )
        node_sum.index_add_(0, endpoint, endpoint_attr)
        node_count.index_add_(
            0,
            endpoint,
            torch.ones(
                endpoint_attr.size(0), 1, device=edge_attr.device, dtype=edge_attr.dtype
            ),
        )
        projected = (projection or self.edge_in)(node_sum / node_count.clamp_min(1.0))
        return projected * (node_count > 0).to(projected.dtype)

    def _bidirectional_edges(
        self, edge_attr: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return receiving node, neighbouring node, and oriented attributes.

        Stored graphs contain one canonical edge per pair.  The AE receives
        normalized edge attributes.  Reversal therefore applies the fitted
        affine offset ``-u - 2*mean/std`` for directional channels, exactly
        matching reversal in raw coordinates followed by normalization.
        """
        row, col = edge_index
        reverse_attr = edge_attr.clone()
        if edge_attr.size(-1) == 4:
            directional_columns = [0, 1]
        elif edge_attr.size(-1) >= 11:
            directional_columns = [0, 1, 2, 3, 9, 10]
        else:
            directional_columns = list(range(min(self.pos_dim, edge_attr.size(-1))))
        if directional_columns:
            offset = self.edge_reverse_offset[directional_columns].to(
                device=edge_attr.device, dtype=edge_attr.dtype
            )
            reverse_attr[:, directional_columns] = (
                -edge_attr[:, directional_columns] + offset
            )
        return (
            torch.cat([col, row]),
            torch.cat([row, col]),
            torch.cat([edge_attr, reverse_attr], dim=0),
        )

    def set_edge_normalization(
        self, edge_mean: Tensor, edge_std: Tensor,
        ref_edge_mean: Tensor | None = None, ref_edge_std: Tensor | None = None,
    ) -> None:
        """Set the exact normalized-vector reversal offset for message layers."""
        if edge_mean.numel() != self.edge_dim or edge_std.numel() != self.edge_dim:
            raise ValueError("Edge normalization dimensionality does not match the AE.")
        self.edge_reverse_offset.copy_(
            (-2 * edge_mean.detach() / edge_std.detach().clamp_min(1e-12)).reshape(-1)
        )
        ref_mean = edge_mean if ref_edge_mean is None else ref_edge_mean
        ref_std = edge_std if ref_edge_std is None else ref_edge_std
        self.ref_edge_reverse_offset.copy_(
            (-2 * ref_mean.detach() / ref_std.detach().clamp_min(1e-12)).reshape(-1)
        )

    def message_pass(self, h: Tensor, edge_attr: Tensor, edge_index: Tensor) -> Tensor:
        """Residual nonlinear neighbour exchange with bounded edge intermediates."""
        if not self.message_mlps or edge_attr.numel() == 0:
            return h
        receiver, neighbour, directed_attr = self._bidirectional_edges(edge_attr, edge_index)
        edge_h = self.message_edge_in(directed_attr)
        for message_mlp, update_mlp, norm in zip(
            self.message_mlps, self.message_updates, self.message_norms
        ):
            summed = torch.zeros_like(h)
            counts = torch.zeros((h.size(0), 1), device=h.device, dtype=h.dtype)
            for start in range(0, receiver.numel(), self.message_chunk_size):
                stop = min(start + self.message_chunk_size, receiver.numel())
                recv = receiver[start:stop]
                msg = message_mlp(
                    torch.cat([h[recv], h[neighbour[start:stop]], edge_h[start:stop]], dim=-1)
                )
                summed.index_add_(0, recv, msg)
                counts.index_add_(0, recv, torch.ones((stop - start, 1), device=h.device, dtype=h.dtype))
            update = update_mlp(torch.cat([h, summed / counts.clamp_min(1)], dim=-1))
            h = norm(h + update)
        return h

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))  # [N, H]
        return self.pool(self.message_pass(h, edge_attr_g, edge_index_g))

    def encode_latent(
        self,
        delta_pos: Tensor,
        h0: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        num_graphs = int(batch.max().item()) + 1
        z_list = []

        for graph_idx in range(num_graphs):
            node_idx = (batch == graph_idx).nonzero(as_tuple=False).flatten()
            delta_pos_g = delta_pos[node_idx]
            h0_g = h0[node_idx]

            global_to_local = torch.full(
                (delta_pos.size(0),),
                -1,
                dtype=torch.long,
                device=batch.device,
            )
            global_to_local[node_idx] = torch.arange(node_idx.numel(), device=batch.device)

            edge_mask = (batch[edge_index[0]] == graph_idx) & (batch[edge_index[1]] == graph_idx)
            edge_index_g = global_to_local[edge_index[:, edge_mask]]
            edge_attr_g = edge_attr[edge_mask]

            z_g = self.encode_latent_graph(delta_pos_g, h0_g, edge_attr_g, edge_index_g)
            z_list.append(z_g)

        return torch.stack(z_list, dim=0)

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        z_tokens = self.decoder_token_proj(z).reshape(
            z.size(0), self.latent_tokens, self.hidden_size
        )
        node_tokens = z_tokens[batch]
        q = self.decoder_query_proj(h0).unsqueeze(1)
        k = self.decoder_key_proj(node_tokens)
        v = self.decoder_value_proj(node_tokens)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_size)
        attn = torch.softmax(scores, dim=-1)
        z_context = (attn.unsqueeze(-1) * v).sum(dim=1)

        return self.node_decoder(torch.cat([z_context, h0], dim=-1))

    def encode(
        self,
        delta_pos: Tensor,
        ref_pos: Tensor,
        edge_attr: Tensor,
        ref_edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h0 = self.encode_reference_graph(ref_pos, ref_edge_attr, edge_index)
        z = self.encode_latent(delta_pos, h0, edge_attr, edge_index, batch)
        return z, h0

    def forward(
        self,
        delta_pos: Tensor,
        ref_pos: Tensor,
        edge_attr: Tensor,
        ref_edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ):
        z, h0 = self.encode(delta_pos, ref_pos, edge_attr, ref_edge_attr, edge_index, batch)
        recon = self.decode(z, h0, batch)
        return recon, z


class NodeDeltaReferenceBottleneckAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Retain a small per-node reference, keeping dynamic/token width unchanged.

    The encoder lifts the retained reference back to dynamic width only while
    combining it with current edge features. The decoder and graph-context
    pool receive exclusively the small reference representation.
    """

    def __init__(self, *, reference_dim: int, **kwargs):
        super().__init__(**kwargs)
        if not 0 < reference_dim < self.hidden_size:
            raise ValueError("Reference bottleneck must be positive and smaller than hidden_size.")
        self.reference_dim = int(reference_dim)
        self.ref_node_in = nn.Linear(self.pos_dim + self.hidden_size, self.reference_dim)
        self.reference_to_dynamic = nn.Linear(self.reference_dim, self.hidden_size)
        self.decoder_query_proj = nn.Linear(self.reference_dim, self.hidden_size)
        self.node_decoder = nn.Sequential(
            nn.Linear(self.hidden_size + self.reference_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.reconstruction_dim),
        )

    def encode_latent_graph(self, delta_pos_g, h0_g, edge_attr_g, edge_index_g):
        return super().encode_latent_graph(
            delta_pos_g, self.reference_to_dynamic(h0_g), edge_attr_g, edge_index_g
        )


class NodeDeltaReference8AutoEncoder(NodeDeltaReferenceBottleneckAutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(reference_dim=8, **kwargs)


class NodeDeltaReference16AutoEncoder(NodeDeltaReferenceBottleneckAutoEncoder):
    def __init__(self, **kwargs):
        super().__init__(reference_dim=16, **kwargs)


class NodeDeltaCorrectedReference16AutoEncoder(NodeDeltaReference16AutoEncoder):
    """Compact reference with reversal consistent with fitted edge statistics."""

    def __init__(self, **kwargs):
        super().__init__(correct_edge_reversal=True, **kwargs)


class NodeDeltaMessagePassingReference16AutoEncoder(NodeDeltaCorrectedReference16AutoEncoder):
    """Neighbour exchange before pooling, retaining only 16 features per node."""

    def __init__(self, *, message_passing_steps: int = 2, **kwargs):
        super().__init__(message_passing_steps=message_passing_steps, **kwargs)


class NodeDeltaSpatialDecoderReference16AutoEncoder(NodeDeltaCorrectedReference16AutoEncoder):
    """Decode with graph-wide node exchange from the compact reference and z.

    Only the 16D reference is retained per node. The wider decoder activations
    are transient, and attention is isolated within each graph in the batch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_decoder = nn.Linear(self.hidden_size + self.reference_dim, self.hidden_size)
        self.spatial_decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=4, dim_feedforward=2*self.hidden_size,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(2)
        ])
        self.spatial_output = nn.Linear(self.hidden_size, self.reconstruction_dim)

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        tokens = self.decoder_token_proj(z).reshape(z.size(0), self.latent_tokens, self.hidden_size)
        node_tokens = tokens[batch]
        q = self.decoder_query_proj(h0).unsqueeze(1)
        k = self.decoder_key_proj(node_tokens)
        v = self.decoder_value_proj(node_tokens)
        attention = torch.softmax((q*k).sum(-1) / math.sqrt(self.hidden_size), dim=-1)
        context = (attention.unsqueeze(-1)*v).sum(1)
        initial = self.node_decoder(torch.cat([context, h0], dim=-1))
        output = initial.new_zeros((h0.size(0), self.reconstruction_dim))
        for graph_idx in range(z.size(0)):
            indices = (batch == graph_idx).nonzero(as_tuple=False).flatten()
            if not indices.numel():
                continue
            state = initial[indices].unsqueeze(0)
            for layer in self.spatial_decoder:
                state = layer(state)
            output = output.index_copy(0, indices, self.spatial_output(state.squeeze(0)))
        return output


class NodeDeltaDirectAttentionAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Attention autoencoder whose decoder directly returns node displacements.

    Each reference-node query attends over latent tokens whose values are
    two-dimensional displacement vectors. The attention-weighted value is the
    node prediction, with no additional node decoder MLP.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_value_proj = nn.Linear(self.hidden_size, self.pos_dim)
        self.node_decoder = None

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        z_tokens = self.decoder_token_proj(z).reshape(
            z.size(0), self.latent_tokens, self.hidden_size
        )
        node_tokens = z_tokens[batch]
        q = self.decoder_query_proj(h0).unsqueeze(1)
        k = self.decoder_key_proj(node_tokens)
        displacement_values = self.decoder_value_proj(node_tokens)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_size)
        attention = torch.softmax(scores, dim=-1)
        return (attention.unsqueeze(-1) * displacement_values).sum(dim=1)


class NodeDeltaSingleStageAttentionAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Reduce node states directly to one scalar per learned latent query."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = DirectLatentAttentionPool(self.hidden_size, self.latent_dim)


class NodeDeltaMessagePassingAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Pyramid-attention AE with residual nonlinear neighbour messages."""

    def __init__(self, *, message_passing_steps: int = 2, **kwargs):
        super().__init__(message_passing_steps=message_passing_steps, correct_edge_reversal=True, **kwargs)


class NodeDeltaOrientationCorrectedAttentionAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Baseline attention AE with raw-coordinate-consistent edge reversal."""

    def __init__(self, **kwargs):
        super().__init__(correct_edge_reversal=True, **kwargs)


class NodeDeltaMLPAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Minimal expressive autoencoder using mean pooling and shallow MLPs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_node_in = nn.Sequential(
            nn.Linear(self.pos_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.node_in = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.pool = None
        self.to_latent = nn.Linear(self.hidden_size, self.latent_dim)
        self.decoder_token_proj = None
        self.decoder_query_proj = None
        self.decoder_key_proj = None
        self.decoder_value_proj = None
        self.decoder_latent_proj = None
        self.node_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.pos_dim),
        )

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))
        return self.to_latent(h.mean(dim=0))

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        return self.node_decoder(torch.cat([z[batch], h0], dim=-1))


class NodeDeltaPyramidMLPAutoEncoder(NodeDeltaMLPAutoEncoder):
    """MLP-weighted N -> 100 -> 20 -> latent_dim pyramid without Q/K attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = PyramidMLPPool(self.hidden_size, self.latent_dim)
        self.to_latent = None
        self.node_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.pos_dim),
        )

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))
        return self.pool(h)
