"""Eq. (32): public distribution, observed dense target, no fitted solution.

Same RandomState draw order and standard-coordinate arithmetic as the authors.
Unused interleaved output and the illustrative plot are omitted.
"""
import numpy as np


def generate(seed):
    k_vectors = np.random.RandomState(int(seed)).normal(0, 1, (30, 3))
    state = np.zeros(2**24, dtype=np.float64)
    for start in range(0, state.size, 2**20):
        end = start + 2**20
        indices = np.arange(start, end, dtype=np.uint32)
        bits = ((indices[:, None] >> np.arange(24)[::-1]) & 1).astype(np.uint8)
        weights = 2 ** np.arange(1, 9)
        x = np.sum(bits[:, :8] / weights, axis=1)
        y = np.sum(bits[:, 8:16] / weights, axis=1)
        z = np.sum(bits[:, 16:] / weights, axis=1)
        for j in range(1, 31):
            state[start:end] += np.cos(j * (k_vectors[j-1, 0]*x + k_vectors[j-1, 1]*y + k_vectors[j-1, 2]*z))
    return state.reshape((2,)*24)
