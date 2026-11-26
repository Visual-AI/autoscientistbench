"""Verbatim notebook SW, 500 unit-normal projections; no rescaling."""
import numpy as np

def wasserstein_1d_sorted(x, y):
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))

    m = min(len(x), len(y))
    if len(x) != len(y):
        q = (np.arange(m) + 0.5) / m
        x = np.quantile(x, q)
        y = np.quantile(y, q)

    return np.mean(np.abs(x - y))

def sliced_wasserstein_distance(X, Y, n_proj=500, seed=123):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    d = X.shape[1]
    rng = np.random.default_rng(seed)

    dirs = rng.normal(size=(n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    sw_vals = np.empty(n_proj, dtype=float)
    for j in range(n_proj):
        v = dirs[j]
        sw_vals[j] = wasserstein_1d_sorted(X @ v, Y @ v)

    return float(np.mean(sw_vals))

