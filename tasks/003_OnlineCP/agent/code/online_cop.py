"""Causal state form of the released COP update, with its original arithmetic."""
import numpy as np


class OnlineTail:
    def __init__(self, lr, burn, alpha=.05, scale=.5, proportional_lr=True):
        self.lr, self.burn = float(lr), int(burn)
        self.alpha, self.scale = float(alpha), float(scale)
        self.proportional_lr = bool(proportional_lr)
        self.scores = []
        self.q = np.float64(0)
        self.qt = np.float64(0)

    def update(self, score):
        t = len(self.scores)
        window = np.asarray(self.scores[max(t - self.burn, 0):t], dtype=np.float64)
        lr_t = self.lr * (window.max() - window.min()) if self.proportional_lr and t > 0 else self.lr
        grad = self.alpha if self.q >= score else -(1 - self.alpha)
        if t < self.burn:
            grad_i = 0
        else:
            grad_i = np.mean(window <= self.qt - lr_t * grad) - (1 - self.alpha)
        integrator = -self.scale * grad_i
        self.qt = self.qt - lr_t * grad
        self.q = self.qt + lr_t * integrator
        self.scores.append(np.float64(score))


class OnlineCOP:
    def __init__(self, lr, burn, alpha=.05, scale=.5, proportional_lr=True):
        self.tails = [OnlineTail(lr, burn, alpha, scale, proportional_lr) for _ in range(2)]

    def predict(self, forecast):
        lower = forecast - self.tails[0].q
        upper = forecast + self.tails[1].q
        # The released runner orders the two endpoints after independent tail updates.
        return float(np.minimum(lower, upper)), float(np.maximum(lower, upper))

    def observe(self, forecast, y):
        self.tails[0].update(forecast - y)
        self.tails[1].update(y - forecast)
