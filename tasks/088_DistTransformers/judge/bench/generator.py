"""Frozen Experiment 4.1 random stream (float32, PyTorch CPU).

Prior parameters are supplied to the inference method; the sampled variance is
latent. Evaluation seeds and posterior sampling state never belong in input.
"""
import hashlib
import torch
from torch.distributions import InverseGamma, MultivariateNormal


def sample_instances(shape):
    a = InverseGamma(4, 6).sample(shape)
    b = InverseGamma(4, 6).sample(shape)
    phi = torch.stack([a, b], dim=-1)
    x = InverseGamma(a, b).sample()
    z = MultivariateNormal(torch.tensor([0.], dtype=torch.float32),
                          covariance_matrix=x.unsqueeze(-1).unsqueeze(-1)).sample()
    return phi, x, z


def generate(seed, n=1000, n_mc=10000):
    # The original driver initializes the model before drawing evaluation data.
    from reference_api import build_model
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        build_model()
        phi, x, z = sample_instances((n,))
        a, b = phi.unbind(-1)
        prior = InverseGamma(a, b)
        prior_samples = prior.sample((n_mc,))
        posterior_rng = torch.get_rng_state()
        post_a, post_b = a + .5, b + z.squeeze(-1).square() / 2
        posterior_samples = InverseGamma(post_a, post_b).sample((n_mc,))
    return dict(phi=phi, x=x, z=z, a=post_a, b=post_b,
                posterior_rng=posterior_rng,
                prior_sample_hash=hashlib.sha256(prior_samples.numpy().tobytes()).hexdigest(),
                posterior_sample_hash=hashlib.sha256(posterior_samples.numpy().tobytes()).hexdigest())
