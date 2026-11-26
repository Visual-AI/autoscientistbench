"""Frozen simulation generator from the reproduction; explicit RNG and coordinate inputs.
Full-tissue generation precedes uniform nested subsampling. DNB unit = 0.5 um.
"""
import hashlib
import numpy as np

PATTERNS = ['frontier'] + [f'square_{w}' for w in (200, 500, 1000, 2000)] + [f'square_random_{w}' for w in (200, 500, 1000, 2000)] + [f'line_{o}_{w}' for o in ('horizontal', 'vertical') for w in (100, 250, 500)]
NULLS = [f'null_{p:02d}' for p in (50, 40, 30, 20, 10, 5)]
LEVELS = [250, 500, 1000, 5000, 10000]


def seed_for(scenario, rep, role='simulation'):
    return int.from_bytes(hashlib.blake2s(f'Bio-34__metric|{scenario}|{rep}|{role}'.encode(), digest_size=4).digest(), 'little')


def generate(coords, scenario, simulation_seed):
    """Generate on ALL S1 cells, before any subsampling. DNB unit = 0.5 um."""
    rng = np.random.default_rng(simulation_seed)
    n = len(coords)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    params = {}
    if scenario.startswith('null_'):
        p = int(scenario.split('_')[1]) / 100
        values = np.zeros(n, dtype=np.int8)
        values[rng.permutation(n)[:round(p*n)]] = 1
        return values, np.zeros(n, dtype=bool), {'fraction_ones': p}
    if scenario == 'frontier':
        theta = rng.uniform(0, 2*np.pi)
        anchor = rng.uniform(lo, hi)
        normal = np.array([np.cos(theta), np.sin(theta)])
        mask = (coords-anchor) @ normal >= 0
        params = {'theta': float(theta), 'anchor_dnb': anchor.tolist()}
    elif scenario.startswith('square'):
        width = 2*int(scenario.split('_')[-1])
        corner = rng.uniform(lo, hi-width)
        mask = ((coords >= corner) & (coords <= corner+width)).all(axis=1)
        params = {'lower_left_dnb': corner.tolist(), 'width_dnb': width}
    elif scenario.startswith('line'):
        _, orientation, width = scenario.split('_')
        width = 2*int(width)
        axis = 1 if orientation == 'horizontal' else 0
        start = rng.uniform(lo[axis], hi[axis]-width)
        mask = (coords[:,axis] >= start) & (coords[:,axis] <= start+width)
        params = {'axis': axis, 'start_dnb': float(start), 'width_dnb': width}
    else:
        raise ValueError(scenario)
    if 'random' in scenario or scenario.startswith('line'):
        values = rng.integers(0, 2, n, dtype=np.int8)
        values[mask] = 1
    else:
        values = mask.astype(np.int8)
    return values, mask, params



def subsample(coords, values, mask, n, subsample_seed):
    sub = np.random.default_rng(subsample_seed).permutation(len(values))[:n]
    return coords[sub], values[sub], mask[sub], sub
