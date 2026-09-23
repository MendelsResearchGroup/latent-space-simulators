"""Node-only local/global cold-network acceleration simulator."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from global_context_simulator import MeshMessageBlock,_mlp
class NodeOnlyGlobalSimulator(nn.Module):
 def __init__(self,hidden_dim=64,atom_embedding_dim=8,attention_heads=4,attention_layers=2,atom_type_count=2):
  super().__init__(); self.atom_embedding=nn.Embedding(atom_type_count,atom_embedding_dim); self.node_encoder=nn.Sequential(nn.Linear(9+atom_embedding_dim,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim)); self.edge_encoder=nn.Sequential(nn.Linear(6,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim)); self.pre=nn.ModuleList(MeshMessageBlock(hidden_dim) for _ in range(2)); layer=nn.TransformerEncoderLayer(d_model=hidden_dim,nhead=attention_heads,dim_feedforward=2*hidden_dim,dropout=0.,activation="gelu",batch_first=True,norm_first=True); self.transformer=nn.TransformerEncoder(layer,num_layers=attention_layers,norm=nn.LayerNorm(hidden_dim)); self.post=nn.ModuleList(MeshMessageBlock(hidden_dim) for _ in range(2)); self.decoder=_mlp(hidden_dim,hidden_dim,3); nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)
 def forward(self,nodes:Tensor,types:Tensor,edges:Tensor,index:Tensor)->Tensor:
  x=self.node_encoder(torch.cat((nodes,self.atom_embedding(types)),1)); e=self.edge_encoder(edges)
  for b in self.pre:x,e=b(x,e,index)
  x=self.transformer(x.unsqueeze(0)).squeeze(0)
  for b in self.post:x,e=b(x,e,index)
  return self.decoder(x)
