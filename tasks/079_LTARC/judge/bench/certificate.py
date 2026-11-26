"""Trusted FFT grid, structure checking and reconstruction; never import candidate code."""
import math
import numpy as np
import pandas as pd

TAU = .35
MAX_SPLITS = 3
MAX_CANDIDATES = 100000  # Output safety bound, not the baseline search size.


def frame(x, names, boolean, a=None, y=None):
    out = pd.DataFrame({name: np.asarray(x[:, j], dtype=bool if boolean[j] else float)
                        for j, name in enumerate(names)})
    if a is not None:
        out.insert(0, 'a', a)
    if y is not None:
        out.insert(1, 'y', y)
    return out


def training_grid(x, names, boolean):
    """Exact released 40-bin percentile grid, order and Boolean conditions."""
    criteria = []
    for feature in range(len(names)):
        values = x[:, feature]
        if boolean[feature]:
            for label in (True, False):
                criteria.append({'feature': feature, 'bins': [True], 'labels': [label]})
        else:
            percentiles = np.linspace(0, 100, 41, endpoint=False)[1:]
            lo, hi = float(np.min(values)), float(np.max(values))
            for point in reversed(np.unique(np.percentile(values, percentiles))):
                if point != lo and point != hi:
                    for labels in ([True, False], [False, True]):
                        criteria.append({'feature': feature, 'bins': [lo, float(point), hi],
                                         'labels': labels})
    return criteria


def binary(value):
    return type(value) is int and value in (0, 1)


def check_tree(tree, criteria):
    if not isinstance(tree, dict) or set(tree) != {'rules', 'default'}:
        raise ValueError('invalid FFT schema')
    if not binary(tree['default']) or not isinstance(tree['rules'], list):
        raise ValueError('invalid FFT leaf')
    if len(tree['rules']) > MAX_SPLITS:
        raise ValueError('FFT exceeds three splits')
    for rule in tree['rules']:
        if not isinstance(rule, dict) or set(rule) != {'criterion', 'decision'}:
            raise ValueError('invalid FFT condition')
        key = rule['criterion']
        if type(key) is not int or not 0 <= key < len(criteria) or not binary(rule['decision']):
            raise ValueError('FFT condition outside training grid')
    return tree


def check_candidates(document, criteria):
    if not isinstance(document, dict) or set(document) != {'candidates'}:
        raise ValueError('invalid candidate document')
    candidates = document['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError('invalid number of candidates')
    for tree in candidates:
        check_tree(tree, criteria)
    return candidates


def selected_tree(document, candidates):
    if not isinstance(document, dict) or set(document) != {'selected'}:
        raise ValueError('certification must select a committed index')
    index = document['selected']
    if type(index) is not int or not 0 <= index < len(candidates):
        raise ValueError('invalid certification selection')
    return candidates[index]


def decisions(tree, x, criteria):
    """Preserve pd.cut edge and original Boolean reverse-condition behavior."""
    check_tree(tree, criteria)
    d = np.full(len(x), tree['default'], dtype=np.int8)
    exited = np.zeros(len(x), dtype=bool)
    for rule in tree['rules']:
        condition = criteria[rule['criterion']]
        values = pd.Series(x[:, condition['feature']])
        if len(condition['bins']) == 3:
            # Out-of-training-range values become NaN then True, exactly as released.
            continuation = pd.cut(values, bins=condition['bins'], include_lowest=True,
                                  labels=condition['labels']).to_numpy(dtype=bool)
        else:
            # Released implementation ignores the reverse label for Boolean criteria.
            continuation = (values == condition['bins'][0]).to_numpy(dtype=bool)
        newly_exited = ~exited & ~continuation
        d[newly_exited] = rule['decision']
        exited |= newly_exited
    return d


def risks(d, a, y):
    if d.shape != a.shape or y.shape != a.shape or len(d) == 0:
        raise ValueError('invalid decision dimensions')
    loss = 2 * y * (d == a)
    population = math.fsum(map(float, loss)) / len(loss)
    treated = int(d.sum())
    treated_loss = math.fsum(map(float, loss[d == 1]))
    treatment = 0.0 if treated == 0 else (treated_loss / treated if treated_loss else 1.0)
    return population, treatment


def aggregate(values, tau=TAU):
    if not values:
        raise ValueError('no repetitions')
    population = math.fsum(p for p, t in values) / len(values)
    treatment = math.fsum(t for p, t in values) / len(values)
    coverage = sum(t <= tau for p, t in values) / len(values)
    passed = coverage >= .90 and treatment <= tau
    return {'population_risk': population if passed else 1.0,
            'population_risk_unguarded': population, 'treatment_risk': treatment,
            'coverage': coverage, 'guard_pass': passed}
