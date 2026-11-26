"""Frozen real-data stream: observe current X, submit estimate, then advance.
No latent means are loaded in this process. Opaque seeds never generate observations.
Private closures are an API boundary, not a security boundary; replay and audit are required.
"""
import csv
import math


def _make_source(observations):
    # Instrumentation seam: the judge replaces this factory to recount reads.
    return iter(tuple(float(x) for x in observations))


class Episode:
    __slots__ = ('horizon', '_observe', '_submit', '_finish')
    def __init__(self, horizon, observe, submit, finish):
        self.horizon = horizon
        self._observe, self._submit, self._finish = observe, submit, finish
    def observe(self):
        return self._observe()
    def submit(self, estimate):
        return self._submit(estimate)
    def finish(self):
        return self._finish()


def make_episode(seed, observations):
    int(seed)  # opaque identifier; real data has no instance or noise RNG
    horizon = len(observations)
    source = _make_source(observations)
    estimates = []
    pending, closed, draws = False, False, 0
    def observe():
        nonlocal pending, draws
        if closed or pending or draws >= horizon:
            raise RuntimeError('submit current estimate before observing again')
        value = next(source)
        draws += 1
        pending = True
        return value
    def submit(estimate):
        nonlocal pending
        if closed or not pending:
            raise RuntimeError('observe once before each submission')
        value = float(estimate)
        if not math.isfinite(value):
            raise ValueError('nonfinite estimate')
        estimates.append(value)
        pending = False
    def finish():
        nonlocal closed
        if closed or pending or len(estimates) != horizon:
            raise RuntimeError('finish exactly once, after all submissions')
        closed = True
        return {'estimates': list(estimates), 'draws': draws}
    return Episode(horizon, observe, submit, finish)


def read_input(path):
    groups = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'episode', 'seed', 't', 'x']:
            raise ValueError('bad input header')
        seen = set()
        for row in reader:
            if row['id'] in seen:
                raise ValueError('duplicate input id')
            seen.add(row['id'])
            g = groups.setdefault(row['episode'], {'seed': int(row['seed']), 'ids': [], 'x': []})
            if int(row['seed']) != g['seed'] or int(row['t']) != len(g['x']) + 1:
                raise ValueError('inconsistent episode or time order')
            value = float(row['x'])
            if not math.isfinite(value):
                raise ValueError('nonfinite observation')
            g['ids'].append(row['id']); g['x'].append(value)
    if not groups:
        raise ValueError('empty input')
    return list(groups.values())
