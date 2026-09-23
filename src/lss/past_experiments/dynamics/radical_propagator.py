"""Archived graph-operator, soft-expert, and conditional-flow study models."""
import torch
from torch import nn
from .trajectory_propagator import ReferenceContext, mlp

FAMILIES = ('graph_operator', 'soft_experts', 'trajectory_flow')


class GraphOperator(nn.Module):
    """Graph-conditioned dissipative linear dynamics in an8D lifted state."""
    def __init__(self):
        super().__init__()
        self.context = ReferenceContext(18,64,'attention')
        self.generator = mlp(66,64,64)
        self.lift = mlp(66,64,8)
        self.readout = nn.Linear(8,2,bias=False)

    def forward(self,z0,tokens):
        condition = torch.cat((self.context(tokens),z0))
        raw = self.generator(condition).reshape(8,8)
        operator = .05*(raw-raw.T)-.1*torch.diag(torch.nn.functional.softplus(raw.diag())+.001)
        initial = self.lift(condition)
        times = torch.arange(101,device=z0.device,dtype=z0.dtype)/100
        states = torch.matrix_exp(times[:,None,None]*operator) @ initial
        return z0+self.readout(states-initial)

    def sample_paths(self,z0,tokens):
        return self(z0,tokens).unsqueeze(0)


class SoftExperts(nn.Module):
    """Four graph/time trajectory experts, routed without source labels."""
    def __init__(self):
        super().__init__()
        self.context = ReferenceContext(18,64,'attention')
        self.gate = mlp(66,64,4)
        self.experts = nn.ModuleList([mlp(67,64,2) for _ in range(4)])

    def forward(self,z0,tokens):
        condition = torch.cat((self.context(tokens),z0))
        times = torch.arange(101,device=z0.device,dtype=z0.dtype).reshape(-1,1)/100
        inputs = torch.cat((condition.expand(101,-1),times),-1)
        weights = self.gate(condition).softmax(-1)
        trajectories = torch.stack([expert(inputs) for expert in self.experts])
        return z0+times*(weights[:,None,None]*trajectories).sum(0)

    def sample_paths(self,z0,tokens):
        return self(z0,tokens).unsqueeze(0)


class TrajectoryFlow(nn.Module):
    """Conditional flow matching over the200-dimensional latent displacement path.

    Inference uses8 fixed antithetic noise samples,16 Euler integration steps.
    Every sample is conditioned solely on the initial latent and reference graph.
    """
    def __init__(self,motion_scale):
        super().__init__()
        self.context = ReferenceContext(18,64,'attention')
        self.velocity = mlp(267,256,200)
        self.register_buffer('motion_scale',motion_scale.reshape(2).clone())
        generator = torch.Generator().manual_seed(91823)
        noise = torch.randn(4,200,generator=generator)
        self.register_buffer('evaluation_noise',torch.cat((noise,-noise)))

    def _field(self,x,time,condition):
        return self.velocity(torch.cat((x,condition.expand(len(x),-1),time.expand(len(x),1)),-1))

    def flow_training(self,z0,tokens,true_path):
        condition = torch.cat((self.context(tokens),z0))
        clean = ((true_path[1:]-z0)/self.motion_scale).reshape(1,200)
        noise = torch.randn_like(clean)
        time = torch.rand((),device=z0.device,dtype=z0.dtype)
        point = (1-time)*noise+time*clean
        velocity = self._field(point,time,condition)
        flow_loss = (velocity-(clean-noise)).square().mean()
        estimate = point+(1-time)*velocity
        path = torch.cat((z0[None],z0+self.motion_scale*estimate.reshape(100,2)))
        return path,flow_loss

    def sample_paths(self,z0,tokens):
        condition = torch.cat((self.context(tokens),z0))
        point = self.evaluation_noise
        for step in range(16):
            time = z0.new_tensor(step/16)
            point = point+self._field(point,time,condition)/16
        future = z0+self.motion_scale*point.reshape(8,100,2)
        return torch.cat((z0.reshape(1,1,2).expand(8,1,2),future),1)

    def forward(self,z0,tokens):
        return self.sample_paths(z0,tokens).mean(0)


def make_radical_propagator(family,motion_scale):
    if family=='graph_operator':
        return GraphOperator()
    if family=='soft_experts':
        return SoftExperts()
    if family=='trajectory_flow':
        return TrajectoryFlow(motion_scale)
    raise ValueError(family)
