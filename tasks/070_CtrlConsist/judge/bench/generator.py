"""Frozen CIR generator: released float32 PRNG split, Euler scan and CIR drift.
Labels are for offline validation only; prediction must use a learned controller.
"""
import jax
import jax.numpy as jnp
from jax import lax
EPS = 1e-12

def get_problem(cfg):
    a = cfg.get("a", 1.0)
    b = cfg.get("b", 1.0)
    sigma = cfg.get("sigma", 1.0)
    d = cfg.get("d", 1)
    T = cfg.get("T", 1.0)
    x_0 = jnp.array(cfg.get("x_0", [2.0]))
    x_T = jnp.array(cfg.get("x_T", [2.0]))

    def base_drift(x, t):
        return a * (b - x)

    def sigma_fn(x, t):
        return sigma * jnp.sqrt(jnp.maximum(x, 0.0))

    def cir_transition_params(x, y, s):
        s = jnp.maximum(s, EPS)
        exp_as = jnp.exp(-a * s)
        c = 2.0 * a / ((1.0 - exp_as) * sigma**2)
        x_clip = jnp.clip(x, 0.0)
        y_clip = jnp.clip(y, 0.0)
        u = c * x_clip * exp_as
        v = c * y_clip
        q = 2.0 * a * b / (sigma**2) - 1.0
        z = 2.0 * jnp.sqrt(jnp.maximum(u * v, EPS))
        return c, u, v, q, z, exp_as

    def I_ratio_cf(q, z, n_terms=500):
        z = jnp.maximum(z, EPS)
        z2 = z * z

        def body(i, r):
            k = n_terms - i
            return z2 / (2.0 * (q + k) + r)

        tail = lax.fori_loop(0, n_terms, body, jnp.zeros_like(z))
        return z / (2.0 * (q + 1.0) + tail)

    def Iprime_over_I(q, z):
        z = jnp.maximum(z, EPS)
        R_true = I_ratio_cf(q, z)
        R_asymp = -(q + 0.5) * (1.0 / z) + 0.5 * (q * q - 0.25) * (1.0 / z**2) + 1.0
        R = jnp.where(z > 1e4, R_asymp, R_true)
        return q / z + R

    def dlogp_dx_CIR(x, y, s):
        c, u, v, q, z, exp_as = cir_transition_params(x, y, s)
        du_dx = c * exp_as
        sqrt_v_over_u = jnp.sqrt(jnp.maximum(v, EPS) / jnp.maximum(u, EPS))
        dlogp_du = (
            -1.0
            - q / (2.0 * jnp.maximum(u, EPS))
            + Iprime_over_I(q, z) * sqrt_v_over_u
        )
        return du_dx * dlogp_du

    def true_drift_fn(x, t):
        return base_drift(x, t) + (sigma**2) * jnp.clip(x, 0.0) * dlogp_dx_CIR(x, x_T, T - t)

    return {
        "base_drift": base_drift,
        "sigma_fn": sigma_fn,
        "shape": (d,),
        "x_0": x_0,
        "x_T": x_T,
        "T": T,
        "true_drift_fn": true_drift_fn,
    }

def _get_sigma_fn(sigma, dim):
    """
    Returns a callable sigma_fn(x,t) -> (apply_sigma, sigma_val)
    where:
      - apply_sigma(dB_t) computes sigma @ dB_t efficiently
    """
    # --- normalize sigma to something callable
    if callable(sigma):
        def base_sigma_val(x,t):
            return sigma(x,t)
    else:
        def base_sigma_val(x,t):
            return sigma

    def sigma_fn(x, t):
        val = jnp.asarray(base_sigma_val(x,t))
        if val.ndim == 0:
            # scalar case: multiply directly
            s = val
            def apply_sigma(dB_t):
                return s * dB_t
            return s, apply_sigma

        elif val.ndim == 1:
            # diagonal case
            if val.shape[0] != dim:
                raise ValueError(f"sigma vector has shape {val.shape}, expected ({dim},)")
            s = val
            def apply_sigma(dB_t):
                return s * dB_t  # elementwise
            return s, apply_sigma

        elif val.ndim == 2:
            # full matrix
            if val.shape != (dim, dim):
                raise ValueError(f"sigma matrix has shape {val.shape}, expected ({dim}, {dim})")
            s = val
            def apply_sigma(dB_t):
                return s @ dB_t
            return s, apply_sigma

        else:
            raise ValueError("sigma must be scalar, vector (len d), or (d,d) matrix.")
        
    def a_inv_fn(x, t, v):
        # applies (sigma sigmaᵀ)^{-1} @ v. Handles scalar, diagonal, and full matrix cases.
        val = jnp.asarray(base_sigma_val(x, t))
        if val.ndim == 0:
            return v / (val**2 + 1e-8)
        elif val.ndim == 1:
            return v / (val**2 + 1e-8)
        else:
            a_matrix = val @ val.T
            eps = 1e-8
            return jnp.linalg.solve(a_matrix + eps*jnp.eye(a_matrix.shape[0]), v)
            # return jnp.linalg.solve(a_matrix, v)
    
    return sigma_fn, a_inv_fn

def euler_maruyama_sampler(
        x,
        t,
        dB_t,
        dt,
        drift_fn,
        sigma_fn,
):
    drift = drift_fn(x, t)
    sigma, apply_sigma_fn = sigma_fn(x, t)
    noise_term = apply_sigma_fn(dB_t)
    x_next = x + drift * dt + noise_term
    return x_next, drift

def _sample_sde_fn(key, drift_fn, sigma_fn, sampler_fn, x_0, shape, ts):
    """
    Samples the SDE for the given drift, using the given sampler function and time discretisation.
    """
    def step(carry, t):
        x, key, prev_t = carry
        key, subkey = jax.random.split(key)
        dt = t - prev_t
        dB_t = jax.random.normal(subkey, shape=shape) * jnp.sqrt(dt)
        x_next, drift = sampler_fn(x, prev_t, dB_t, dt, drift_fn, sigma_fn)
        output = (x_next, drift, dB_t)
        return (x_next, key, t), output

    carry = (x_0, key, ts[0])
    _, outputs = jax.lax.scan(step, carry, ts[1:])
    traj, drift_traj, dBt_traj = outputs
    
    traj = jnp.vstack([x_0[None, :], traj])
    drift_traj = jnp.vstack([drift_traj, drift_traj[-1][None, :]])   # repeat the final one, to make a consistent size. It is not used in the computation.
    dBt_traj = jnp.vstack([dBt_traj, dBt_traj[-1][None, :]])

    return traj, drift_traj, dBt_traj

def generate(seed):
    problem = get_problem({})
    ts = jnp.linspace(0, 1.0, 501)
    sigma, _ = _get_sigma_fn(problem['sigma_fn'], 1)
    def sample(key):
        return _sample_sde_fn(key, problem['true_drift_fn'], sigma,
                              euler_maruyama_sampler, problem['x_0'], (1,), ts)
    xs, drift, noise = jax.jit(jax.vmap(sample))(jax.random.split(jax.random.PRNGKey(seed), 1000))
    sigmas = jax.vmap(jax.vmap(problem['sigma_fn'], in_axes=(0,0)), in_axes=(0,None))(xs[:,:-1],ts[:-1])
    return xs, drift[:,:-1], noise, sigmas, ts
