#!/usr/bin/env python3
"""The single configuration reproduced here: Section 4.2 / Table 1 dense coefficient setting.

Circulant symmetric covariance Sigma_cs, Gaussian noise N(0,1), MAR mechanism (23),
dense beta_0^(de), query point x^(3) = e_100.

Index conventions (i = query point, k = coefficient vector) follow the authors' scripts
Paper_Code/Cir_Sym/Debiased_Inf_CirSym.py exactly.
"""
import numpy as np

D = 1000
N = 900
RHO = 0.1          # circulant symmetric covariance, Javanmard & Montanari (2014) Sec 5.1
X_INDEX = 2        # x^(3) = e_100
BETA_INDEX = 1     # beta_j proportional to 1/sqrt(j), norm 5
MECHANISM = "MAR"


def sigma_circulant(d=D, rho=RHO):
    Sigma = np.zeros((d, d)) + np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            if (j < i + 6) or (j > i + d - 6):
                Sigma[i, j] = rho
                Sigma[j, i] = rho
    return Sigma


def query_point(d=D, i=X_INDEX):
    x = np.zeros(d)
    if i == 0:
        x[0] = 1
    elif i == 1:
        x[0], x[1], x[2], x[6], x[7] = 1, 1 / 2, 1 / 4, 1 / 2, 1 / 8
    elif i == 2:
        x[99] = 1
    elif i == 3:
        x = 1 / np.linspace(1, d, d)
    elif i == 4:
        x = 1 / np.linspace(1, d, d) ** 2
    elif i == 5:
        x = np.ones(d) / np.sqrt(d)
    else:
        raise ValueError(i)
    return x


def beta0(d=D, k=BETA_INDEX):
    if k == 0:
        b = np.zeros(d)
        b[:5] = np.sqrt(5)
    elif k == 1:
        b = 1 / np.sqrt(np.linspace(1, d, d))
        b = 5 * b / np.linalg.norm(b)
    elif k == 2:
        b = 1 / np.linspace(1, d, d)
        b = 5 * b / np.linalg.norm(b)
    else:
        raise ValueError(k)
    return b


def simulate(seed, Sigma=None, d=D, n=N):
    """Data generation of the authors' script (RNG call order preserved)."""
    if Sigma is None:
        Sigma = sigma_circulant(d)
    beta_0 = beta0(d)
    np.random.seed(seed)
    X_sim = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    eps_err_sim = 1.0 * np.random.randn(n)
    Y_sim = np.dot(X_sim, beta_0) + eps_err_sim
    R1 = np.random.choice([0, 1], size=n, replace=True, p=[0.3, 0.7])
    obs_prob2 = 1 / (1 + np.exp(-1 + X_sim[:, 6] - X_sim[:, 7]))
    R2 = np.ones((n,))
    R2[np.random.rand(n) >= obs_prob2] = 0
    return X_sim, Y_sim, R1, R2, beta_0
