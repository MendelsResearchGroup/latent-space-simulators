"""Contracts for the reduced structural representation and its saved models."""
import io
import pytest
import torch
from lss.latent.experiment import _autoencoder_class
from lss.latent.models import DeltaLatentDynamicsMLP


@pytest.mark.parametrize('width',[8,16])
@pytest.mark.parametrize('latent_dim',[2,4])
def test_small_reference_is_used_by_encoder_decoder_and_context(width,latent_dim):
    torch.manual_seed(52)
    cls=_autoencoder_class(f'attention_reference{width}')
    kwargs=dict(pos_dim=2,edge_dim=4,hidden_size=96,latent_dim=latent_dim,latent_tokens=32)
    model=cls(**kwargs)
    x=torch.randn(5,2,requires_grad=True)
    edges=torch.tensor([[0,1,2,3],[1,2,3,4]])
    edgeattr=torch.randn(4,4,requires_grad=True)
    batch=torch.zeros(5,dtype=torch.long)
    h0=model.encode_reference_graph(x,edgeattr,edges)
    assert h0.shape==(5,width)
    z=model.encode_latent(torch.zeros_like(x),h0,edgeattr,edges,batch)
    pred=model.decode(z,h0,batch)
    assert z.shape==(1,latent_dim) and pred.shape==(5,2)
    assert model.decoder_query_proj.in_features==width
    assert model.node_decoder[0].in_features==96+width
    with_context=DeltaLatentDynamicsMLP(latent_dim,64,context_dim=width,graph_context_dim=16)
    no_context=DeltaLatentDynamicsMLP(latent_dim,64,context_dim=0)
    delta=with_context(z,h0.mean(0,keepdim=True)) + no_context(z)
    (pred.square().mean()+delta.square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert model.reference_to_dynamic.weight.grad.abs().sum()>0
    assert no_context.net[0].in_features==latent_dim
    assert no_context.context_dim==0
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    restored=cls(**kwargs);restored.load_state_dict(torch.load(buffer,weights_only=True))
    h1=restored.encode_reference_graph(x.detach(),edgeattr.detach(),edges)
    z1=restored.encode_latent(torch.zeros_like(x),h1,edgeattr.detach(),edges,batch)
    torch.testing.assert_close(restored.decode(z1,h1,batch),pred.detach())


def test_baseline_remains_96_wide_without_a_new_lift():
    m=_autoencoder_class('attention')(pos_dim=2,edge_dim=4,hidden_size=96,latent_dim=2,latent_tokens=32)
    assert m.ref_node_in.out_features==96
    assert m.decoder_query_proj.in_features==96
    assert not hasattr(m,'reference_to_dynamic')


@pytest.mark.parametrize('name,steps',[
    ('attention_reference16_corrected',0),('message_passing_reference16',2)])
def test_compact_lj_models_preserve_edge_orientation_and_reload(name,steps):
    torch.manual_seed(61)
    cls=_autoencoder_class(name)
    kwargs=dict(pos_dim=2,edge_dim=5,hidden_size=24,latent_dim=8,
                latent_tokens=4,message_passing_steps=steps)
    model=cls(**kwargs)
    mean=torch.tensor([.5,-.3,1.,2.,.2])
    std=torch.tensor([.7,.4,.5,1.2,.3])
    model.set_edge_normalization(mean,std)
    raw=torch.tensor([[1.,.1,1.005,2.,0.],[.2,.4,.447,0.,1.],
                      [.5,-.4,.640,1.,0.]])
    edges=torch.tensor([[0,0,1],[1,2,2]])
    attr=(raw-mean)/std
    ref=torch.randn(3,2)
    delta=torch.randn(3,2,requires_grad=True)
    batch=torch.zeros(3,dtype=torch.long)
    pred,z=model(delta,ref,attr,attr,edges,batch)
    reversed_raw=raw.clone();reversed_raw[:,:2]*=-1
    rev=(reversed_raw-mean)/std
    pred_rev,z_rev=model(delta,ref,rev,rev,edges.flip(0),batch)
    torch.testing.assert_close(pred,pred_rev,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(z,z_rev,atol=2e-6,rtol=2e-5)
    assert model.encode_reference_graph(ref,attr,edges).shape==(3,16)
    (pred.square().mean()+z.square().mean()).backward()
    assert torch.isfinite(delta.grad).all() and delta.grad.abs().sum()>0
    if steps:
        assert model.message_mlps[0][0].weight.grad.abs().sum()>0
    restored=cls(**kwargs);restored.load_state_dict(model.state_dict())
    # Derived reversal offsets are restored from the bundle's normalizers.
    restored.set_edge_normalization(mean,std)
    pred2,z2=restored(delta.detach(),ref,attr,attr,edges,batch)
    torch.testing.assert_close(pred,pred2)
    torch.testing.assert_close(z,z2)
