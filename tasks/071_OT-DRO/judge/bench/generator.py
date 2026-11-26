#!/usr/bin/env python3
"""Frozen data-generating process of the task (paper Sec. 6.2 / App. E.2, Table 3, J = 10).

An *instance* is identified by (model_id, seed):
  * model_id in 0..9 selects one of the ten true linear models (w_m, sigma_m^2) of the paper's Table 3
    (the authors drew them once with w ~ U(-10, 10), sigma^2 ~ U(500, 1000) and then kept them fixed);
  * seed drives numpy's default_rng and produces J = 10 observations x_j ~ U(-10, 10),
    y_j = w_m x_j + e_j, e_j ~ N(0, sigma_m^2), followed by n_b = 20 bootstrap resamples of the 10 points
    (index arrays, drawn with replacement).  The draw order (x, then noise, then the 20 index arrays) is
    exactly the one of the archived reproduction, so (model_id, seed) reproduces its instances bit for bit.

The *observed* part of an instance (x, y, bootstrap_indices) is what `predict.sh` receives.  The true
parameters (w_m, sigma_m^2) are labels; so is the seed of the M = 5000-point sample of the true distribution
that the scorer uses as the empirical proxy of P for the coverage guard (`true_sample`).

val = public seeds (materialised in data/val), test = hidden seeds held by the judge.  This file is frozen:
`predict.sh` may import it (BENCH_BENCH_DIR) but must not use the true parameters of an instance.
"""
import numpy as np

# Table 3 of the paper: (w, sigma^2), rows 1..10 -> model_id 0..9
MODELS = [
    (-6.7805, 564.285), (-5.8464, 625.412), (-2.7811, 699.653), (-1.3851, 710.190), (-0.0144, 783.458),
    (4.3483, 846.372), (6.3163, 915.492), (7.1061, 922.537), (8.5174, 932.399), (8.9350, 978.001),
]
J = 10                 # observations per dataset (Sec. 6.2 multi-experiment, Fig. 5 right / Fig. 7 at J = 10)
N_BOOTSTRAP = 20       # n_b bootstrap resamples (Sec. 4 / Table 1)
X_LOW, X_HIGH = -10.0, 10.0
COVERAGE_M = 5000      # size of the empirical proxy of the true distribution used by the coverage guard


def generate_instance(model_id, seed):
    """-> dict(x (J,), y (J,), bootstrap_indices (N_BOOTSTRAP, J) int, w_true, sigma2)."""
    w, var = MODELS[int(model_id)]
    rng = np.random.default_rng(int(seed))
    X = rng.uniform(X_LOW, X_HIGH, (J, 1))
    y = np.dot(X, np.array([w])) + rng.normal(0.0, np.sqrt(var), J)
    idx = np.array([rng.choice(J, J, replace=True) for _ in range(N_BOOTSTRAP)])
    return {"x": X[:, 0], "y": y, "bootstrap_indices": idx, "w_true": float(w), "sigma2": float(var)}


def true_sample(model_id, coverage_seed, m=COVERAGE_M):
    """M i.i.d. points of the true P of `model_id` as a discrete distribution (support (M, 2), uniform weights)."""
    w, var = MODELS[int(model_id)]
    rng = np.random.default_rng(int(coverage_seed))
    xx = rng.uniform(X_LOW, X_HIGH, (m, 1))
    yy = w * xx[:, 0] + rng.normal(0.0, np.sqrt(var), m)
    return np.column_stack((xx[:, 0], yy)), np.full(m, 1.0 / m)


def noise_floor(model_id):
    """E|e| = sigma sqrt(2/pi): the best possible e_oos of model `model_id` (attained at w = w_true)."""
    return float(np.sqrt(MODELS[int(model_id)][1]) * np.sqrt(2.0 / np.pi))


def materialize(ids, model_ids, seeds, coverage_seeds, input_path, truth_path, extra=None):
    """Write the observed part to input_path (.npz) and everything else to truth_path (.npz).

    input : id (n,) str, x (n, J), y (n, J), bootstrap_indices (n, N_BOOTSTRAP, J)
    truth : id, model_id, w_true, sigma2, seed, coverage_seed, x, y, bootstrap_indices [+ extra arrays]
    """
    inst = [generate_instance(m, s) for m, s in zip(model_ids, seeds)]
    obs = {"id": np.array(ids, dtype=str),
           "x": np.stack([d["x"] for d in inst]), "y": np.stack([d["y"] for d in inst]),
           "bootstrap_indices": np.stack([d["bootstrap_indices"] for d in inst]).astype(np.int64)}
    truth = dict(obs, model_id=np.array(model_ids, dtype=np.int64), seed=np.array(seeds, dtype=np.int64),
                 coverage_seed=np.array(coverage_seeds, dtype=np.int64),
                 w_true=np.array([d["w_true"] for d in inst]), sigma2=np.array([d["sigma2"] for d in inst]))
    truth.update(extra or {})
    np.savez(input_path, **obs)
    np.savez(truth_path, **truth)
    return obs, truth


if __name__ == "__main__":
    import sys
    m, s = int(sys.argv[1]), int(sys.argv[2])
    d = generate_instance(m, s)
    print({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()})
