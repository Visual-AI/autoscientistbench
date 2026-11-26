"""Frozen binary DGP. Legacy RandomState and author draw order are preserved.

This module describes the distribution, not an authorized source of extra data.
Use only supplied observations in train/predict. Seeds/parameters are never inputs.
"""
import numpy as np

MECHANISMS = ('no_bias', 'transportability_bias', 'confounding_bias', 'selection_bias')
KEYS = ('Y0', 'Y1', 'A', 'S')


def sample_probs(rng, d, low, high, biased):
    p = [rng.choice([rng.uniform(*low), rng.uniform(*high)]) for _ in range(2**d)]
    if not biased:
        p[1::2] = p[::2]
    return np.asarray(p)


def cohort(rng, n, r, probs, transport=False):
    x = rng.choice([0, 1], size=(n, 5), p=([.6, .4] if r else [.4, .6]))
    group = x @ (2**np.arange(4, -1, -1))
    if transport:
        u_probs = sample_probs(rng, 5, (.1, .9), (.1, .9), True)
        u = rng.binomial(1, u_probs[group])
    else:
        u_probs = np.full(32, .5)
        u = rng.choice([0, 1], size=(n, 1), p=[.5, .5]).reshape(-1)
    full_group = group*2 + u
    values = {key: rng.binomial(1, probs[key][full_group]) for key in KEYS}
    # The input is ONLY measured covariates, selection and factual selected outcomes.
    a, s = values['A'], values['S']
    y = a*values['Y1'] + (1-a)*values['Y0']
    observed = np.column_stack((x, s, np.where(s == 1, a, -1), np.where(s == 1, y, -1))).astype(np.int8)
    # Hidden draws are returned separately for operator verification and truth storage.
    hidden = dict(u=u, u_probs=u_probs, **values)
    return observed, hidden


def generate(seed, mechanism, count):
    """Yield complete independent experiments in the original sequential stream.

    count controls the INITIAL draw of p_bounds and thus must match the source
    batch even when retaining a prefix. LogisticRegression(lbfgs) consumes no RNG.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(mechanism)
    rng = np.random.RandomState(seed)
    bounds = rng.uniform(.2, .5, count)
    flags = (False, mechanism != 'no_bias', mechanism == 'confounding_bias', mechanism == 'selection_bias')
    for bound in bounds:
        obs = {key: sample_probs(rng, 6, (.1, bound), (1-bound, .9), flag) for key, flag in zip(KEYS, flags)}
        rct = dict(Y0=obs['Y0'], Y1=obs['Y1'], A=np.full(64, .5), S=np.full(64, .95))
        cohorts, aux = [], []
        for n, r, probs in ((50000, 1, rct), (50000, 0, obs), (40000, 1, rct), (40000, 0, obs)):
            a, h = cohort(rng, n, r, probs, mechanism == 'transportability_bias' and not r)
            cohorts.append(a); aux.append(h)
        yield np.concatenate(cohorts), dict(bound=bound, obs_probs=np.asarray(list(obs.values())),
                                         rct_probs=np.asarray(list(rct.values())), auxiliary=aux)
