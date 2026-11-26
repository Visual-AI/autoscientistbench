"""Frozen original nonparanormal draw order and preprocessing.

Supply a precision matrix and seed. True parameters are returned only for data
materialization and diagnostics. Prediction may not consult them or simulate
additional data. The original model is fixed; truth-table use is prohibited.
"""
import numpy as np
import torch


def generate_instance(theta, seed):
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        prec=torch.as_tensor(theta,dtype=torch.float32)
        sigma=torch.inverse(prec)
        dist=torch.distributions.MultivariateNormal(torch.zeros(50),sigma)
        raw=dist.sample((2048,))
        auxiliary=dist.sample((1000,))
        x=torch.sign(raw)*torch.abs(raw).pow(3.0)
        xt=torch.sign(auxiliary)*torch.abs(auxiliary).pow(3.0)
        mean=x.mean(dim=0,keepdim=True); std=x.std(dim=0,keepdim=True)
        x=(x-mean)/(std+1e-6);xt=(xt-mean)/(std+1e-6)
        truth=sigma.inverse().numpy();np.fill_diagonal(truth,0)
        truth[np.abs(truth)<1e-5]=0
        return dict(x=x.numpy(),auxiliary=xt.numpy(),theta=prec.numpy(),sigma=sigma.numpy(),true_prec=truth,rng_state=torch.get_rng_state().numpy())
