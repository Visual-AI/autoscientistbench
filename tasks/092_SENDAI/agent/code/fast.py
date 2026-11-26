"""Algebraically factor the INR's first affine map; preserve every dropout call."""
import torch
import torch.nn.functional as F

def decode_at_coords(self, z, coords):
    batch_size, n = z.shape[0], coords.shape[0]
    pe = self.pe(coords)
    first = self.decoder[0]
    k = pe.shape[-1]
    pos = F.linear(pe, first.weight[:, :k], first.bias)
    latent = F.linear(z, first.weight[:, k:])
    x = (pos.unsqueeze(0) + latent.unsqueeze(1)).reshape(batch_size*n, -1)
    for layer in list(self.decoder.children())[1:]:
        x = layer(x)
    return x.reshape(batch_size, n)

def install():
    from SENDAI.models import HF_SHRED_INR
    HF_SHRED_INR.decode_at_coords = decode_at_coords

_original_dropout_forward = torch.nn.Dropout.forward

def exact_dropout_forward(self, x):
    if not self.training or self.p == 0:
        return x
    if x.device.type != 'cpu' or x.dtype != torch.float32 or not x.is_contiguous() or x.numel()<4096 or self.p==1:
        return _original_dropout_forward(self,x)
    noise = torch.ops.sendai_opt30.mask(x.detach(), 1-self.p)
    noise.div_(1-self.p)
    return x.mul_(noise) if self.inplace else x*noise

def install_rng():
    import os
    from pathlib import Path
    torch.ops.load_library(str(Path(os.environ['SENDAI_BUILD_DIR'])/'sendai_rng.so'))
    torch.nn.Dropout.forward = exact_dropout_forward

def previous_hf_for_sensors(layer, residual):
    """Training uses only sensor entries of frozen previous HF outputs."""
    if not layer.training:
        return layer(residual)
    assert not torch.is_grad_enabled()
    z=layer.encode(residual);batch=z.shape[0];idx=layer.sensor_indices;k=idx.numel()
    pe=layer.pe(layer.coord_grid[idx]);first=layer.decoder[0];npe=pe.shape[-1]
    x=(F.linear(pe,first.weight[:,:npe],first.bias).unsqueeze(0)+F.linear(z,first.weight[:,npe:]).unsqueeze(1)).reshape(batch*k,-1)
    for op in list(layer.decoder.children())[1:]:
        if isinstance(op,torch.nn.Dropout):
            shaped=x.reshape(batch,k,-1)
            noise=torch.ops.sendai_opt30.selected_mask(shaped,layer.state_dim,idx,1-op.p)
            x=x*noise.div_(1-op.p).reshape_as(x)
        else:
            x=op(x)
    out=x.new_zeros((batch,layer.state_dim))
    out[:,idx]=layer.scale*x.reshape(batch,k)
    return out

def install_previous():
    import inspect
    import SENDAI.training as training
    training.previous_hf_for_sensors=previous_hf_for_sensors
    for name in ['train_hf_layer','train_hf_finetune']:
        source=inspect.getsource(getattr(training,name))
        source=source.replace('full_model.hf_layers[i](res)','previous_hf_for_sensors(full_model.hf_layers[i], res)')
        exec(compile(source,'<opt30 '+name+'>','exec'),training.__dict__)

