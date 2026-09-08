import torch
from lss.latent.experiment import _autoencoder_class


def test_spatial_decoder_equivariance_batch_isolation_and_gradients():
    torch.manual_seed(128)
    cls=_autoencoder_class('attention_reference16_spatial_decoder')
    kwargs=dict(pos_dim=2,edge_dim=5,hidden_size=24,latent_dim=4,latent_tokens=4)
    model=cls(**kwargs)
    model.eval()
    z=torch.randn(2,4,requires_grad=True)
    h=torch.randn(7,16,requires_grad=True)
    batch=torch.tensor([0,0,0,1,1,1,1])
    out=model.decode(z,h,batch)
    individual=torch.cat([model.decode(z[i:i+1],h[batch==i],torch.zeros(int((batch==i).sum()),dtype=torch.long)) for i in range(2)])
    torch.testing.assert_close(out,individual,atol=1e-6,rtol=1e-5)
    order=torch.tensor([5,0,6,2,3,1,4])
    torch.testing.assert_close(model.decode(z,h[order],batch[order]),out[order],atol=1e-6,rtol=1e-5)
    out.square().sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert torch.isfinite(h.grad).all() and h.grad.abs().sum()>0
    assert model.spatial_decoder[0].self_attn.in_proj_weight.grad.abs().sum()>0
    restored=cls(**kwargs);restored.eval();restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.decode(z.detach(),h.detach(),batch),out.detach())
