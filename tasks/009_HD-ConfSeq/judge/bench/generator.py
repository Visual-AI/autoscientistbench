"""Frozen author data generators. Parameters are public distributions, never instance identities."""
from typing import *
import re
from enum import Enum
import os
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX512F,AVX2,FMA3")
import numpy as np
from scipy.special import ndtr
from scipy.stats import beta as sp_beta
HIGHLY_CORRELATED_RHO=0.98

def get_batched_ab_stream(
        T: int = 200,
        n_per_arm: int = 500,
        seed: int = 0,
        noise_dims: int = 2,
        # Which (named) metrics are truly affected by treatment:
        effect_metrics: Sequence[str] = ("time", "clicks", "scroll", "add_to_cart"),
        # Strength is on the *latent engagement shift* used only for affected metrics.
        # Values around 0.03–0.08 are reasonable; larger => quicker detection.
        engagement_shift: float = 0.06,
        variable_batch_sizes: bool = True,
        min_batch: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
        Z: (T, D) array in [0,1], where Z_t = 0.5 * ( (mean_B - mean_A) + 1 ).
           Null corresponds to 0.5 per coordinate.
        meta: dict with metric names, batch sizes, and diagnostic arrays.

    Dimensions:
        Base metrics: 8
        + noise_dims (default 2) => D = 10 by default.
    """

    rng = np.random.default_rng(seed)

    # ---------- Metric configuration ----------
    # Caps for clipping (raw -> bounded)
    CAPS = {
        "time_sec": 300.0,
        "clicks": 20.0,
        "pages": 10.0,
    }

    base_metric_names = [
        "time",  # scaled time_sec / 300
        "clicks",  # scaled clicks / 20
        "pages",  # scaled pages / 10
        "scroll",  # already in [0,1]
        "add_to_cart",  # in {0,1}
        "purchase",  # in {0,1}
        "non_bounce",  # in {0,1}, equals 1 - bounce
        "return_7d",  # in {0,1}
    ]
    noise_metric_names = [f"noise_{j + 1}" for j in range(noise_dims)]
    metric_names = base_metric_names + noise_metric_names

    name_to_idx = {n: i for i, n in enumerate(base_metric_names)}

    # Which base metrics get treatment effect
    effect_idx = [name_to_idx[m] for m in effect_metrics if m in name_to_idx]

    D_base = len(base_metric_names)
    D = D_base + noise_dims

    # ---------- Session simulator ----------
    def _simulate_sessions(n: int, arm: str) -> np.ndarray:
        """
        Simulate n sessions for arm in {"A","B"}.
        Returns scaled session-level X in [0,1]^(D_base + noise_dims).
        """
        # Latent engagement drives correlation across metrics
        # (Beta chosen to produce many low-engagement, few high-engagement sessions)
        engagement = rng.beta(2.0, 5.0, size=n)  # in [0,1]

        # For arm B, only *affected metrics* use a shifted engagement signal.
        # This keeps some coordinates close to null while others move.
        if arm == "B" and engagement_shift != 0:
            engagement_eff = np.clip(engagement + engagement_shift, 0.0, 1.0)
        else:
            engagement_eff = engagement

        # Helper: pick engagement per metric (shifted only for affected metrics)
        def e_for_metric(metric_index: int) -> np.ndarray:
            if metric_index in effect_idx:
                return engagement_eff
            return engagement

        # --- Raw metrics on natural-ish scales ---
        # time_sec: lognormal noise around an engagement-dependent mean
        e_time = e_for_metric(name_to_idx["time"])
        base_time = 20.0 + 220.0 * e_time  # seconds
        time_noise = rng.lognormal(mean=0.0, sigma=0.35, size=n)  # mean ~ 1.06
        time_sec = base_time * time_noise
        time_sec = np.clip(time_sec, 0.0, CAPS["time_sec"])

        # clicks: Poisson with engagement-dependent rate
        e_clicks = e_for_metric(name_to_idx["clicks"])
        lam_clicks = 0.5 + 8.0 * e_clicks
        clicks = rng.poisson(lam=lam_clicks, size=n).astype(float)
        clicks = np.clip(clicks, 0.0, CAPS["clicks"])

        # pages: Poisson with engagement-dependent rate
        e_pages = e_for_metric(name_to_idx["pages"])
        lam_pages = 0.8 + 4.5 * e_pages
        pages = rng.poisson(lam=lam_pages, size=n).astype(float)
        pages = np.clip(pages, 0.0, CAPS["pages"])

        # scroll: Beta distribution pushed higher with engagement
        e_scroll = e_for_metric(name_to_idx["scroll"])
        a = 1.5 + 8.0 * e_scroll
        b = 2.5 + 3.0 * (1.0 - e_scroll)
        scroll = rng.beta(a, b)

        # add_to_cart: Bernoulli with engagement-dependent prob
        e_atc = e_for_metric(name_to_idx["add_to_cart"])
        p_atc = np.clip(0.03 + 0.40 * e_atc, 0.0, 1.0)
        add_to_cart = rng.binomial(1, p_atc, size=n).astype(float)

        # purchase: depends on engagement and add_to_cart
        e_buy = e_for_metric(name_to_idx["purchase"])
        p_buy = np.clip(0.005 + 0.22 * e_buy * add_to_cart, 0.0, 1.0)
        purchase = rng.binomial(1, p_buy, size=n).astype(float)

        # bounce: higher when engagement is low; we store non_bounce = 1 - bounce
        # Keep bounce unaffected unless you include "non_bounce" in effect_metrics.
        e_bounce = e_for_metric(name_to_idx["non_bounce"])
        p_bounce = np.clip(0.60 - 0.45 * e_bounce, 0.0, 1.0)
        bounce = rng.binomial(1, p_bounce, size=n).astype(float)
        non_bounce = 1.0 - bounce

        # return_7d: engagement-dependent probability
        e_ret = e_for_metric(name_to_idx["return_7d"])
        p_ret = np.clip(0.08 + 0.30 * e_ret, 0.0, 1.0)
        return_7d = rng.binomial(1, p_ret, size=n).astype(float)

        # --- Scale to [0,1] ---
        time = time_sec / CAPS["time_sec"]
        clicks_s = clicks / CAPS["clicks"]
        pages_s = pages / CAPS["pages"]

        X_base = np.column_stack([
            time,
            clicks_s,
            pages_s,
            scroll,
            add_to_cart,
            purchase,
            non_bounce,
            return_7d,
        ]).astype(np.float64)

        # Noise dimensions: i.i.d. Uniform(0,1), no treatment effect by construction
        if noise_dims > 0:
            X_noise = rng.uniform(0.0, 1.0, size=(n, noise_dims))
            X = np.concatenate([X_base, X_noise], axis=1)
        else:
            X = X_base

        # Safety: enforce bounds numerically
        X = np.clip(X, 0.0, 1.0)
        return X

    # ---------- Batched pairing ----------
    Z = np.zeros((T, D), dtype=np.float64)
    mean_A = np.zeros((T, D), dtype=np.float64)
    mean_B = np.zeros((T, D), dtype=np.float64)
    batch_sizes = np.zeros((T, 2), dtype=int)  # (nA, nB)

    for t in range(T):
        if variable_batch_sizes:
            # +/- 15% jitter, but never below min_batch
            jitterA = int(rng.integers(-int(0.15 * n_per_arm), int(0.15 * n_per_arm) + 1))
            jitterB = int(rng.integers(-int(0.15 * n_per_arm), int(0.15 * n_per_arm) + 1))
            nA = max(min_batch, n_per_arm + jitterA)
            nB = max(min_batch, n_per_arm + jitterB)
        else:
            nA = nB = n_per_arm

        XA = _simulate_sessions(nA, arm="A") # returns (nA, D)
        XB = _simulate_sessions(nB, arm="B")

        mA = XA.mean(axis=0)
        mB = XB.mean(axis=0)

        delta = mB - mA  # in [-1,1]^D
        z = 0.5 * (delta + 1.0)  # in [0,1]^D

        mean_A[t] = mA
        mean_B[t] = mB
        Z[t] = np.clip(z, 0.0, 1.0)
        batch_sizes[t] = (nA, nB)

    meta = {
        "metric_names": metric_names,
        "base_metric_names": base_metric_names,
        "noise_dims": noise_dims,
        "effect_metrics": list(effect_metrics),
        "effect_indices_base": effect_idx,
        "engagement_shift": engagement_shift,
        "caps": CAPS,
        "batch_sizes": batch_sizes,  # shape (T,2)
        "mean_A": mean_A,  # shape (T,D)
        "mean_B": mean_B,  # shape (T,D)
        "delta": mean_B - mean_A,  # shape (T,D)
        "Z_definition": "Z_t = 0.5 * ((mean_B - mean_A) + 1), null at 0.5",
    }
    return Z.astype(np.float64), meta

def generate_samples(D: int, N: int, radii_per_dim: np.ndarray,
                     center: np.ndarray, pow: int = 2) -> (
        Tuple)[np.ndarray, np.ndarray]:
    assert pow >= 2 and pow % 2 == 0, "Only even values of p >= 2 are supported."

    assert pow > 0, "pow must be positive"
    radii_per_dim = np.asarray(radii_per_dim).reshape(1, D)
    center = np.asarray(center).reshape(1, D)

    # 1) sample directions on the L_pow sphere
    if pow == 2:
        # special-case: standard normal -> directions on Euclidean sphere (fast)
        vecs = np.random.randn(N, D)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
    else:
        # general-case: sample coordinates X with density ~ exp(-|x|^p)
        # |X|^p ~ Gamma(shape=1/p, scale=1), sign random
        shape = 1.0 / float(pow)
        T = np.random.gamma(shape=shape, scale=1.0, size=(N, D))  # T = |X|^p
        R = T ** (1.0 / float(pow))  # |X|
        signs = np.random.choice([-1.0, 1.0], size=(N, D))
        vecs = signs * R
        # normalize by L_p norm to lie on the L_p *sphere*
        lp_norms = (np.sum(np.abs(vecs) ** float(pow), axis=1, keepdims=True)) ** (1.0 / float(pow))
        vecs = vecs / lp_norms

    # 2) uniform radius in the ball (volume scales as r^D for any p)
    base_radii = np.random.rand(N) ** (1.0 / D)  # in [0,1], density ~ r^{D-1}
    scaled_dirs = vecs * base_radii[:, None]

    # 3) scale per-dimension by radii and translate by center
    samples = center + scaled_dirs * radii_per_dim

    true_mean = center.flatten()
    return samples, true_mean

class DistributionType(Enum):
    UNIFORM = "uniform"
    UNIFORM2 = "uniform2"
    GAUSSIAN = "gaussian"
    BETA = "beta"
    DIRICHLET = "dirichlet"
    FIXED = "fixed"

    # higher dimensional
    FIXED_ND = "fixed_nd"
    UNIFORM_ND = "uniform_nd"  # Doesn't sum to 1
    HIGHLY_CORRELATED_ND = "highly_correlated_nd"  # Doesn't sum to 1
    CIRCLE_ND = "circle_nd"  # points on the surface of a unit circle
    ELLIPSE_ND = "ellipse_nd"  # points on the surface of an ellipse
    BERNOULLI_ND = "bernoulli_nd"  # Bernoulli samples with p=0.5 in {0, 1}
    BERNOULLI_MIX_ND = "bernoulli_mix_nd"

class DataDistributionChecker:
    @staticmethod
    def dim(h: dict) -> int:
        """Return the number of output dimensions for the distribution described by h."""
        distribution_type, args, dim = DataDistributionChecker.check_distribution(h["data_type"])
        return dim

    @staticmethod
    def check_distribution(type_str: str) -> Tuple[DistributionType, Any, int]:
        """Parse a distribution description string and return a tuple
        ret: (DistributionType, args, dim)
        """

        if type_str == "uniform":
            return DistributionType.UNIFORM, [0.5, 0.8], 1

        if type_str == "uniform2":
            return DistributionType.UNIFORM2, [0.0, 1.0], 1

        if type_str == "fixed":
            return DistributionType.FIXED, [0.7], 1

        # fixed_Nd: matches e.g. "fixed_3d" -> d=3
        m = re.match(r"^fixed_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            # create a mean vector that sums to 1 with a dominant first dimension
            if nb_dims == 1:
                mean_vector = [0.7]
            else:
                mean_dim1 = 0.7
                remaining = (1.0 - mean_dim1) / max(1, nb_dims - 1)
                mean_vector = [mean_dim1] + [remaining] * (nb_dims - 1)
            return DistributionType.FIXED_ND, mean_vector, nb_dims

        if type_str == "gaussian":
            return DistributionType.GAUSSIAN, [0.0, 1.0], 1

        # beta: beta_a=1.0_b=2.0  (both floats)
        m = re.match(r"^beta_a=(?P<a>-?[0-9.]+(?:\.[0-9]+)?)_b=(?P<b>-?[0-9.]+(?:\.[0-9]+)?)$", type_str)
        if m:
            a = float(m.group("a"))
            b = float(m.group("b"))
            return DistributionType.BETA, [a, b], 1

        # dirichlet with repetition: dirichlet_a=0.5_0.3*10  -> base=0.5_0.3, multiplier=10
        m = re.match(r"^dirichlet_a=(?P<base>[0-9._-]+)\*(?P<mult>\d+)$", type_str)
        if m:
            base_values = m.group("base")
            multiplier = int(m.group("mult"))
            a = [float(x) for x in base_values.split("_")]
            a = a * multiplier
            return DistributionType.DIRICHLET, a, len(a)

        # dirichlet without repetition: dirichlet_a=0.5_0.3_0.2
        m = re.match(r"^dirichlet_a=(?P<vals>[0-9._-]+)$", type_str)
        if m:
            a = [float(x) for x in m.group("vals").split("_")]
            return DistributionType.DIRICHLET, a, len(a)

        # highly_correlated_Nd: highly_correlated_3d
        m = re.match(r"^highly_correlated_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            mean_vector = [0.5] * nb_dims
            return DistributionType.HIGHLY_CORRELATED_ND, mean_vector, nb_dims

        # uniform_nd: support both "uniform_nd_3d" and "uniform_3d" for compatibility
        m = re.match(r"^uniform(?:_nd)?_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            # we return the dimension as args (keeps compatibility with the old code)
            return DistributionType.UNIFORM_ND, nb_dims, nb_dims

        # circle_nd: circle_3d
        m = re.match(r"^circle_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.CIRCLE_ND, nb_dims, nb_dims

        # ellipse_nd: ellipse_3d
        m = re.match(r"^ellipse_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.ELLIPSE_ND, nb_dims, nb_dims

        # bernoulli_nd: bernoulli_3d
        m = re.match(r"^bernoulli_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.BERNOULLI_ND, nb_dims, nb_dims

        # bernoulli_mix_nd: bernoulli_mix_3d
        m = re.match(r"^bernoulli_mix_(?P<d>\d+)d$", type_str)
        if m:
            nb_dims = int(m.group("d"))
            return DistributionType.BERNOULLI_MIX_ND, nb_dims, nb_dims

        raise ValueError(f"Unknown distribution type: {type_str}")

    @staticmethod
    def get_data(type_str: str, N: int) -> Tuple[np.ndarray, Any]:
        """Generate N samples according to the distribution described by type_str.

        Returns (samples, true_mean).
        """
        distribution_type, args, dim = DataDistributionChecker.check_distribution(type_str)

        if distribution_type == DistributionType.UNIFORM:
            lo, hi = args[0], args[1]
            true_mean = 0.5 * (lo + hi)
            samples = np.random.rand(N) * (hi - lo) + lo
            return samples, true_mean

        if distribution_type == DistributionType.UNIFORM2:
            # uniform on [0,1]
            true_mean = 0.5
            return np.random.rand(N), true_mean

        if distribution_type == DistributionType.FIXED:
            val = args[0]
            return np.full(N, val), val

        if distribution_type == DistributionType.FIXED_ND:
            # args is the mean vector
            mean_vector = np.asarray(args, dtype=float)
            samples = np.tile(mean_vector, (N, 1))
            true_mean = mean_vector
            return samples, true_mean

        if distribution_type == DistributionType.GAUSSIAN:
            mu, sigma = args[0], args[1]
            return np.random.normal(loc=mu, scale=sigma, size=N), mu

        if distribution_type == DistributionType.BETA:
            a, b = float(args[0]), float(args[1])
            true_mean = a / (a + b)
            data = sp_beta.rvs(a=a, b=b, size=N)
            np.random.shuffle(data)
            return data, true_mean

        if distribution_type == DistributionType.DIRICHLET:
            a = list(args)
            data = np.random.dirichlet(a, size=N)
            true_mean = np.array(a) / float(np.sum(a))
            # shuffle along the first axis if needed
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            return data[idx], true_mean

        if distribution_type == DistributionType.HIGHLY_CORRELATED_ND:
            # Gaussian copula with equicorrelated latent variables.  Applying
            # the standard-normal CDF leaves every coordinate marginally
            # Uniform[0, 1], while rho close to one concentrates observations
            # tightly around the diagonal.
            mean_vector = np.asarray(args, dtype=float)
            common = np.random.randn(N, 1)
            independent = np.random.randn(N, dim)
            latent = (
                np.sqrt(HIGHLY_CORRELATED_RHO) * common
                + np.sqrt(1.0 - HIGHLY_CORRELATED_RHO) * independent
            )
            samples = ndtr(latent)
            true_mean = mean_vector
            return samples, true_mean

        if distribution_type == DistributionType.UNIFORM_ND:
            d = int(args)
            samples = np.random.rand(N, d)
            true_mean = np.ones(d) * 0.5
            return samples, true_mean

        if distribution_type == DistributionType.CIRCLE_ND or distribution_type == DistributionType.ELLIPSE_ND:
            D = int(args)

            samples, true_mean = generate_samples(
                D, N,
                radii_per_dim=np.array([0.5] + [0.25 / 2] * (D - 1))
                if distribution_type == DistributionType.ELLIPSE_ND else np.full(D, 0.5),
                center=np.full(D, 0.5)
            )

            return samples, true_mean

        if distribution_type == DistributionType.BERNOULLI_ND:
            d = int(args)
            # Bernoulli samples with p=0.5, giving values in {0, 1}
            samples = np.random.binomial(n=1, p=0.5, size=(N, d)).astype(float)
            true_mean = np.ones(d) * 0.5
            return samples, true_mean

        if distribution_type == DistributionType.BERNOULLI_MIX_ND:
            d = int(args)
            samples = np.zeros((N, d))
            # First half: standard Bernoulli(p=0.5) in {0, 1}
            half_d = d // 2
            samples[:, :half_d] = np.random.binomial(n=1, p=0.5, size=(N, half_d)).astype(float)
            # Second half: Bernoulli(p=0.5) shifted to [0.5, 1]
            # If 0 -> 0.5, if 1 -> 1.0
            samples[:, half_d:] = (np.random.binomial(n=1, p=0.5, size=(N, d - half_d)).astype(float) + 1.0) / 2.0
            # True mean: 0.5 for first half, 0.75 for second half
            true_mean = np.concatenate([np.ones(half_d) * 0.5, np.ones(d - half_d) * 0.75])
            return samples, true_mean

        raise ValueError(f"Unknown data type: {type_str}")

    @staticmethod
    def supports_optimistic_rescaling(type_str: str) -> bool:
        """Returns True if the distribution type supports optimistic rescaling."""
        distribution_type, args, dim = DataDistributionChecker.check_distribution(type_str)

        if distribution_type in {
            DistributionType.FIXED_ND,
            DistributionType.CIRCLE_ND,
            DistributionType.ELLIPSE_ND,
        }:
            return True

        return False

    @staticmethod
    def is_legal_distribution(type_str: str) -> bool:
        """Returns True if the distribution type is legal/supported."""
        try:
            DataDistributionChecker.check_distribution(type_str)
            return True
        except ValueError:
            return False

