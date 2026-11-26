"""Verbatim notebook cell 9 fitting and sampling functions."""
import numpy as np
from TAVIE import TAVIE_loc_scale

def sample_normal_gamma_z(m, V, a, b, n_mc, rng):
    m = np.asarray(m, dtype=float).reshape(-1)
    V = np.asarray(V, dtype=float)
    V = 0.5 * (V + V.T)

    jitter = 1e-10
    while True:
        try:
            L = np.linalg.cholesky(V + jitter * np.eye(V.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
            if jitter > 1e-3:
                raise

    tau2 = rng.gamma(shape=a / 2.0, scale=2.0 / b, size=n_mc)
    z_std = rng.normal(size=(n_mc, len(m)))
    beta = m[None, :] + (z_std @ L.T) / np.sqrt(tau2)[:, None]
    log_tau2 = np.log(tau2)
    return np.column_stack([beta, log_tau2])

def fit_tavie_student_and_sample_z(X, y, nu, alpha, n_samples, seed=123):
    p_local = X.shape[1]
    prior_params = [np.zeros(p_local + 1), 10.0 * np.eye(p_local + 1), 0.005, 0.005]

    model = TAVIE_loc_scale(family="student", fit_intercept=True)
    model.fit(X, y, nu=nu, alpha=alpha, prior_params=prior_params, verbose=False)
    ve = model.get_variational_estimates()

    m_xi = np.asarray(ve["m_xi"], dtype=float)
    V_xi = np.asarray(ve["V_xi"], dtype=float)
    a_xi = float(ve["a_xi"])
    b_xi = float(ve["b_xi"])

    rng = np.random.default_rng(seed)
    return sample_normal_gamma_z(m_xi, V_xi, a_xi, b_xi, n_samples, rng)

