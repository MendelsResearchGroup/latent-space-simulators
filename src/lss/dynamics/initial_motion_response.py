"""Initial-motion static-graph encoder for short-horizon global box response."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from global_context_simulator import MeshMessageBlock, _mlp

class InitialMotionBoxResponse(nn.Module):
    def __init__(self, *, hidden_dim: int = 64, attention_heads: int = 4, attention_layers: int = 2, atom_type_count: int = 2, atom_embedding_dim: int = 8):
        super().__init__()
        self.atom_embedding=nn.Embedding(atom_type_count,atom_embedding_dim)
        self.node_encoder=nn.Sequential(nn.Linear(6+atom_embedding_dim,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim))
        self.edge_encoder=nn.Sequential(nn.Linear(6,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim))
        self.blocks=nn.ModuleList(MeshMessageBlock(hidden_dim) for _ in range(2))
        layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=attention_heads,dim_feedforward=2*hidden_dim,dropout=0.,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=attention_layers,norm=nn.LayerNorm(hidden_dim))
        self.global_token=nn.Parameter(torch.zeros(1,1,hidden_dim)); self.decoder=_mlp(hidden_dim+7,hidden_dim,2)
    def encode(self,node_features: Tensor,atom_types: Tensor,edge_features: Tensor,edge_index: Tensor)->Tensor:
        nodes=self.node_encoder(torch.cat((node_features,self.atom_embedding(atom_types)),dim=-1)); edges=self.edge_encoder(edge_features)
        for block in self.blocks: nodes,edges=block(nodes,edges,edge_index)
        return self.transformer(torch.cat((self.global_token,nodes.unsqueeze(0)),dim=1))[0,0]
    def decode(self,embedding: Tensor,context: Tensor)->Tensor:
        if context.ndim!=2 or context.size(1)!=7: raise ValueError("context must be [targets, 7].")
        return self.decoder(torch.cat((embedding.expand(len(context),-1),context),dim=1))
