"""Frozen reproduction metric; no interpolation before error. Truth diagonal is zero before degree sorting."""
import numpy as np

def _deg_sort(W):
    order = np.argsort(np.mean(W, axis=1))
    return W[order][:, order]

def mise_step(Wk, W_true):
    """Exact L2 error of the k x k step-function estimate against the 1000 x 1000 truth."""
    size = W_true.shape[0]
    k = Wk.shape[0]
    Wk_s, Wt_s = _deg_sort(Wk), _deg_sort(W_true)
    b = np.minimum((np.arange(size) * k) // size, k - 1)
    diff = Wk_s[np.ix_(b, b)] - Wt_s
    m = ~np.eye(size, dtype=bool)
    return float(np.mean(diff[m] ** 2))
