"""Trusted single-cell forecast -> interval commitment -> outcome feedback."""
import math
import numpy as np


class Replay:
    def __init__(self, forecast, y):
        self.forecast, self.y = forecast, y
        self.index = 0
        self.waiting = False
        self.failed = False
        self.lower, self.upper = [], []

    def __call__(self, method, args, kwargs):
        try:
            if self.failed or kwargs:
                raise ValueError('invalid temporal request')
            if method == 'next' and not args:
                if self.waiting:
                    raise ValueError('commit before advancing')
                if self.index == len(self.y):
                    return {'op': 'done'}
                self.waiting = True
                return {'op': 'predict', 't': self.index,
                        'forecast': float(self.forecast[self.index])}
            if method != 'submit' or len(args) != 1 or not self.waiting:
                raise ValueError('unexpected temporal operation')
            value = args[0]
            if not isinstance(value, dict) or set(value) != {'t', 'lower', 'upper'}:
                raise ValueError('invalid interval response')
            if type(value['t']) is not int or value['t'] != self.index:
                raise ValueError('wrong time index')
            lo, hi = value['lower'], value['upper']
            if (type(lo) not in (int, float) or type(hi) not in (int, float)
                    or not math.isfinite(lo) or not math.isfinite(hi)
                    or lo > hi or not math.isfinite(hi - lo)):
                raise ValueError('invalid interval bounds')
            # Commit first. The returned outcome cannot affect this interval.
            self.lower.append(float(lo))
            self.upper.append(float(hi))
            t = self.index
            self.index += 1
            self.waiting = False
            return {'op': 'feedback', 't': t, 'y': float(self.y[t])}
        except Exception:
            self.failed = True
            raise

    def intervals(self):
        if self.failed or self.waiting or self.index != len(self.y):
            raise ValueError('incomplete causal transcript')
        return np.asarray(self.lower), np.asarray(self.upper)
