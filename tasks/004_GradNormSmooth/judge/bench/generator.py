"""Frozen notebook-family generator; no evaluation seeds are embedded here."""
import numpy as np
from scipy.special import softmax, logsumexp

def generate(seed):
    rng = np.random.RandomState(seed)
    A = rng.rand(500, 1000) * 2 - 1
    b = rng.rand(1000) * 2 - 1
    return A, b

def centered(A, b):
    # Exactly create_log_sum_exp_zero_oracle(A.T, b, 1.0).
    g = A.dot(softmax(-b))
    return A.T - g

def objective(A, b, x):
    return float(logsumexp(centered(A, b).dot(x) - b))
