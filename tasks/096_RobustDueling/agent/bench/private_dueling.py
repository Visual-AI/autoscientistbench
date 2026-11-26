"""Frozen online environment. Only make_episode and Episode public methods are allowed."""
import numpy as np

T, D, E, K = 2000, 10, 20, 10


def _make_instance_rng(seed):
    return np.random.RandomState(int(seed))


def _make_noise_rng(seed):
    return np.random.RandomState((int(seed) + 172903) % (2**32))


def _instance(seed):
    rng = _make_instance_rng(seed)
    theta = rng.randn(D) * .1
    zeta = rng.randn(E) * 5.
    X = rng.uniform(-np.pi, np.pi, size=(T, K, D))
    Y = np.concatenate([np.cos(X), np.sin(X)], axis=2)
    # Same per-round operations as the released environment (no changed BLAS batching).
    U = np.stack([x @ theta + y @ zeta for x, y in zip(X, Y)])
    return X, Y, U, theta, zeta


class Episode:
    __slots__ = ('_context', '_serve', '_advance', '_finish')
    def get_context(self):
        return self._context()
    def serve(self, a_idx, b_idx):
        return self._serve(a_idx, b_idx)
    def advance(self):
        return self._advance()
    def finish(self):
        return self._finish()


def make_episode(seed):
    X, Y, U, theta, zeta = _instance(seed)
    noise = _make_noise_rng(seed)
    t, corruption_count, regret = 0, 0, 0.
    phase = 'context'
    actions, queue = [], []
    def context():
        nonlocal phase
        if phase != 'context' or t >= T:
            raise RuntimeError('One context per round; serve and advance before the next context')
        phase = 'serve'
        return X[t].copy()
    def serve(a_idx, b_idx):
        nonlocal phase, corruption_count, regret
        if phase != 'serve':
            raise RuntimeError('Serve exactly once after get_context')
        if type(a_idx) not in (int, np.int64, np.int32) or type(b_idx) not in (int,np.int64,np.int32):
            raise ValueError('Arm indices must be integers')
        a, b = int(a_idx), int(b_idx)
        if not (0 <= a < K and 0 <= b < K):
            raise ValueError('Invalid arm')
        ua, ub = U[t,a], U[t,b]
        outcome = int(noise.rand() < 1/(1+np.exp(-(ua-ub))))
        corrupted = corruption_count < 25 and ((ua > ub and outcome == 1) or (ub > ua and outcome == 0))
        if corrupted:
            outcome = 1-outcome
            corruption_count += 1
            delay = 0
        else:
            delay = int(max(0, np.round(noise.normal(100., 100.))))
        queue.append((t+delay,t,outcome))
        best = U[t].max()
        regret += .5 * ((best-ua)+(best-ub))
        actions.append([a,b])
        phase = 'advance'
        return Y[t,a].copy(), Y[t,b].copy()
    def advance():
        nonlocal phase,t,queue
        if phase != 'advance':
            raise RuntimeError('Advance only after serving and observing post-serving context')
        arrived = [(s,o) for arrival,s,o in queue if arrival <= t]
        queue = [q for q in queue if q[0] > t]
        t += 1
        phase = 'context'
        return arrived
    def finish():
        nonlocal phase
        if t != T or phase != 'context':
            raise RuntimeError('finish requires the full horizon and can be called once')
        phase = 'closed'
        return {'actions': [a.copy() for a in actions], 'rounds': t, 'regret': float(regret)}
    ep=Episode()
    ep._context,ep._serve,ep._advance,ep._finish=context,serve,advance,finish
    return ep


class EpisodeBatch:
    """Vector-shaped public API across independent episodes; no temporal lookahead."""
    __slots__ = ('_episodes',)
    def __init__(self, episodes):
        self._episodes = episodes
    def get_context(self):
        return np.stack([ep.get_context() for ep in self._episodes])
    def serve(self, a_indices, b_indices):
        if len(a_indices) != len(self._episodes) or len(b_indices) != len(self._episodes):
            raise ValueError('One pair per episode required')
        values = [ep.serve(a,b) for ep,a,b in zip(self._episodes,a_indices,b_indices)]
        return np.stack([v[0] for v in values]), np.stack([v[1] for v in values])
    def advance(self):
        return [ep.advance() for ep in self._episodes]
    def finish(self):
        return [ep.finish() for ep in self._episodes]


def make_episodes(seeds):
    """Batch factory; same causal methods with an extra leading episode axis."""
    return EpisodeBatch([make_episode(s) for s in seeds])
