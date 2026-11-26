import numpy as np
import torch

def add_gaussian_noise(X, sigma):
    """Add Gaussian noise to the input."""
    noise = torch.normal(mean=0.0, std=sigma, size=X.size(), device=X.device, dtype=X.dtype)
    return X + noise

def normalize_weight(weights, p):

    ws = np.array(weights, dtype=np.float64)
    mu = ws.mean()
    f = 1 / (1 + np.exp(-p * (ws - mu)))

    return f / f.max()

def normalize_features(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x
