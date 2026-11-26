"""CPU batching of the exact portfolio membership predicate (no interpolation)."""
import math
import numpy as np


class PortfolioMembership:
    def __init__(self, calculators, weights, t, threshold):
        self.calculators = calculators
        self.weights = np.asarray(weights)
        self.t = t
        self.threshold = threshold

    def __call__(self, point):
        wealth = np.asarray([calc.get_wealth(self.t, float(m))
                             for calc, m in zip(self.calculators, point)])
        return (float(np.dot(self.weights, wealth)) < self.threshold
                and np.all(point >= 0.0) and np.all(point <= 1.0))

    def _wealth(self, calc, candidates):
        if not all(hasattr(calc, key) for key in ('_advance_from_last_to','_xs','_lams')):
            return np.array([calc.get_wealth(self.t,float(m)) for m in candidates])
        calc._advance_from_last_to(self.t)
        n = self.t * calc.batch_size
        if n == 0:
            return np.full(len(candidates),max(calc.theta,1-calc.theta))
        x = calc._xs[:n][None,:]
        lam = calc._lams[:n][None,:]
        m = candidates[:,None]
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            plus = 1 + np.minimum(lam,calc.c/(m+1e-12)) * (x-m)
            log_plus = np.sum(np.where(plus > 0,np.log(plus),-np.inf),axis=1)
            del plus
            minus = 1 - np.minimum(lam,calc.c/(1-m+1e-12)) * (x-m)
            log_minus = np.sum(np.where(minus > 0,np.log(minus),-np.inf),axis=1)
            log_wealth = np.maximum(math.log(calc.theta)+log_plus,
                                    math.log(1-calc.theta)+log_minus)
            value = np.exp(log_wealth)
            value[log_wealth >= math.log(np.finfo(float).max)] = np.inf
        return value

    def evaluate_many(self, samples, chunk_size=256):
        samples = np.asarray(samples)
        result = np.zeros(len(samples),dtype=bool)
        for start in range(0,len(samples),chunk_size):
            points = samples[start:start+chunk_size]
            active = np.flatnonzero(np.all((points >= 0) & (points <= 1),axis=1))
            wealths = np.zeros((len(points),len(self.weights)))
            partial = np.zeros(len(points))
            for d,calc in enumerate(self.calculators):
                if len(active) == 0: break
                wealths[active,d] = self._wealth(calc,points[active,d])
                partial[active] += self.weights[d] * wealths[active,d]
                # All remaining terms are nonnegative. A numerical margin keeps
                # threshold-near points for the complete scalar predicate below.
                active = active[partial[active] <= self.threshold*(1+1e-12)]
            if len(active):
                total = wealths[active] @ self.weights
                accepted = total < self.threshold
                near = np.abs(total-self.threshold) <= self.threshold*1e-10
                for j in np.flatnonzero(near):
                    accepted[j] = self(points[active[j]])
                result[start+active] = accepted
        return result
