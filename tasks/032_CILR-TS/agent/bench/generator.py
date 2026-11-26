"""Frozen dynamic DGP; same RandomState stream and draw order as DDFMSynthetic.

Only observations are admissible prediction inputs. Seeds and latent arrays are
labels. Generating extra instances or searching for evaluation seeds is forbidden.
"""
import numpy as np


def generate_instance(seed):
    rng = np.random.RandomState(int(seed))
    f = np.zeros((200, 5))
    shocks = rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(200, 5))
    A = 0.7 * np.eye(5)
    for t in range(200):
        f[t] = shocks[t] if t == 0 else f[t - 1] @ A + shocks[t]
    loadings = rng.randn(20, 5)
    eps = np.zeros((200, 20))
    v = rng.randn(200, 20)
    for t in range(200):
        eps[t] = v[t] if t == 0 else 0.2 * eps[t - 1] + v[t]
    return dict(x=f @ loadings.T + eps, factors=f, loadings=loadings,
                shocks=shocks, noise_draws=v, noise=eps)
