"""Frozen observation generator; original legacy MT19937 and Beta draw order."""
import numpy as np

NS = np.linspace(100, 2000, 10, dtype=int)
BS = np.linspace(19, 20.8, 10)

def observations(seed, n, b):
    # An integer reproduces the original counter-seeded simulation exactly.
    # A uint32 seed vector uses the same legacy generator without a small seed space.
    return np.random.RandomState(seed).beta(20, b, int(n))

def archive_instance(i, j, rep):
    seed = (i * 10 + j) * 500 + rep
    x = observations(seed, NS[i], BS[j])
    return x, seed
