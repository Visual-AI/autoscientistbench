"""Frozen causal Figure-1 environment. Only public Episode methods may be used.
The instance seed is an opaque token; algorithm code must not derive state from it.
"""
import csv
import json
import numpy as np
from scipy import stats

N, K, T = 10, 4, 100
F, G, D = N**2, np.sqrt(4*N), np.sqrt(2)


def _make_instance_rng(seed):
    return np.random.RandomState(int(seed))


def _make_noise_rng(seed):
    return np.random.RandomState((int(seed) + 6078001) % (2**32))


def _instance(seed):
    rng, noise = _make_instance_rng(seed), _make_noise_rng(seed)
    Hs, Ys, As, Bs = [], [], [], []
    for _ in range(T):
        h = rng.uniform(-1.0, 1.0, (K, N))
        y = h @ np.ones((N, 1))
        y = y + stats.truncnorm(-1., 1.).rvs(K, random_state=noise).reshape(-1, 1)
        a = rng.uniform(0.0, 2.0, (K, N))
        b = rng.uniform(0.0, 1.0, (K, 1))
        Hs.append(h); Ys.append(y); As.append(a); Bs.append(b)
    x0 = rng.uniform(0., 1., N).reshape(-1, 1)
    return tuple(np.asarray(v) for v in (Hs, Ys, As, Bs)), x0


def _round_metrics(data, x):
    H, y, A, b = data
    loss = 1/2 * np.linalg.norm(H @ x - y)**2
    violation = A @ x - b
    g = max(violation[np.argmax(violation)][0], 0.0)
    return np.asarray([loss, g, g**2], dtype=float)


class Episode:
    __slots__ = ('_initial', '_step', '_finish', '_rounds')

    def __init__(self, initial, step, finish, rounds):
        self._initial, self._step, self._finish, self._rounds = initial, step, finish, rounds

    def initial(self):
        """Return a copy of the public random starting point in [0,1]^N."""
        return self._initial()

    def step(self, x):
        """Commit x_t, then reveal (H_t,y_t,A_t,b_t). No future feedback is exposed."""
        return self._step(x)

    def step_many(self, xs):
        """Commit an already chosen batch in chronological order, then return its feedback."""
        return [self._step(x) for x in xs]

    @property
    def rounds(self):
        return self._rounds()

    def finish(self, final_x):
        """Submit the last unscored iterate after all rounds; return trajectory and counters."""
        return self._finish(final_x)


def make_episode(seed):
    data, x0 = _instance(seed)
    t, closed = 0, False
    totals = np.zeros(3)
    actions = []

    def check_x(x):
        a = np.asarray(x, dtype=float).reshape(N, 1).copy()
        if not np.isfinite(a).all() or np.any(a < -1e-6) or np.any(a > 1+1e-6):
            raise ValueError('decision must lie in [0,1]^N (solver tolerance 1e-6)')
        return a

    def initial():
        return x0.copy()

    def step(x):
        nonlocal t
        if closed or t >= T:
            raise RuntimeError('episode has no remaining feedback')
        x = check_x(x)
        actions.append(x.flatten().tolist())
        row = tuple(v[t] for v in data)
        totals[:] += _round_metrics(row, x)
        t += 1
        return tuple(v.copy() for v in row)

    def finish(final_x):
        nonlocal closed
        if closed or t != T:
            raise RuntimeError('finish requires exactly T committed decisions')
        final_x = check_x(final_x)
        closed = True
        return dict(actions=actions + [final_x.flatten().tolist()], rounds=t,
                    loss=float(totals[0]), ccv1=float(totals[1]), ccv2=float(totals[2]))

    return Episode(initial, step, finish, lambda: t)


HEADER = ['id', 'actions', 'rounds', 'loss', 'ccv1', 'ccv2']


def read_input(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id', 'seed']:
            raise ValueError('input header must be id,seed')
        return [(r['id'], int(r['seed'])) for r in reader]


def write_output(path, records):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(HEADER)
        for key, rec in records:
            w.writerow([key, json.dumps(rec['actions'], separators=(',', ':')), rec['rounds'],
                        repr(rec['loss']), repr(rec['ccv1']), repr(rec['ccv2'])])
