"""Frozen author DGP, exact legacy NumPy stream and draw order.
Only offline materialization may call this generator. Predict consumes observations.
"""
import numpy as np
from scipy.special import expit

def generate(seed, n=100, p=1000, s=0.2, r=10):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 1))
    B = 1.0 * rng.binomial(1, s, size=(1, p))
    U = X @ rng.uniform(size=(1, r), low=-1, high=1) + 0.8 * rng.normal(size=(n, r))
    BU = rng.uniform(size=(r, p), low=-1, high=1) / np.sqrt(r)
    func = lambda u: 1.0 * (BU.T @ u)
    Theta = X @ B + np.apply_along_axis(func, 1, U)
    prob = expit(Theta)
    Y = rng.binomial(1, prob)
    C = rng.choice(np.where(B[0, :] == 0)[0], int(p / 2), replace=False)
    return dict(Y=Y, X=X, U=U, C=C, B=B)
