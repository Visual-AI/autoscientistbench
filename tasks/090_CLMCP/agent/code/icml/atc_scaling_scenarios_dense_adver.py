"""Released ATC exact scan; plotting and unrelated experiment drivers removed."""
import math
from typing import Callable, Dict, Any
import numpy as np

def alpha_r_from_restart(r: int, alpha: float) -> float:
    """Per-restart alpha schedule: alpha_r = (6/pi^2) * alpha / (r+1)^2"""
    return (6.0 / (math.pi**2)) * float(alpha) / ((r + 1) ** 2)

def gamma_paper(t_minus_r: int, sigma: float, alpha_r: float) -> float:
    """
    Paper threshold:
      gamma_t^r = sigma * sqrt( 6 log(t-r) + 2 log(1/alpha_r) + 2 log(pi^2/3) )
    """
    const_term = 2.0 * math.log((math.pi**2) / 3.0)
    return float(sigma) * math.sqrt(
        6.0 * math.log(t_minus_r)
        + 2.0 * math.log(1.0 / alpha_r)
        + const_term
    )

def candidate_splits_all(r: int, t: int) -> np.ndarray:
    """Exact scan: all k in {r+1, ..., t-1}."""
    return np.arange(r + 1, t, dtype=int)

def scan_statistic_max(
    cs: np.ndarray,  # cumulative sum with cs[0]=0, cs[t]=sum_{i=1..t} X_i
    r: int,
    t: int,
    split_fn: Callable[[int, int], np.ndarray],
) -> float:
    """
    Returns C_t^r = max_{k in candidates} sqrt((k-r)(t-k)/(t-r)) * |mean(r+1:k)-mean(k+1:t)|
    """
    k = split_fn(r, t)
    if k.size == 0:
        return 0.0

    len1 = k - r
    len2 = t - k

    valid = (len1 > 0) & (len2 > 0)
    if not np.any(valid):
        return 0.0
    k = k[valid]
    len1 = len1[valid]
    len2 = len2[valid]

    mean1 = (cs[k] - cs[r]) / len1
    mean2 = (cs[t] - cs[k]) / len2
    vals = np.sqrt(len1 * len2 / (t - r)) * np.abs(mean1 - mean2)
    return float(np.max(vals))

def atc_run(
    episode,
    *,
    sigma: float,
    alpha: float,
    split_fn: Callable[[int, int], np.ndarray],
    gamma_fn: Callable[[int, float, float], float],
) -> Dict[str, Any]:
    """
    ATC core:
      alpha_r = (6/pi^2) * alpha / (r+1)^2
      gamma_t^r = gamma_fn(t-r, sigma, alpha_r)
      alarm when C_t^r >= gamma_t^r, then restart r <- t
    """
    T = episode.horizon
    cs = np.zeros(T + 1)

    r = 0
    alpha_r = alpha_r_from_restart(r, alpha)

    hatmu = np.empty(T + 1)
    alarms = []

    for t in range(1, T + 1):
        xt = episode.observe()
        cs[t] = cs[t - 1] + xt
        hatmu[t] = (cs[t] - cs[r]) / (t - r)

        if t >= r + 2:
            C = scan_statistic_max(cs, r, t, split_fn)
            gamma = float(gamma_fn(t - r, sigma, alpha_r))
            if C >= gamma:
                alarms.append(t)
                # Restart at time t: use only X_t for the estimate at this step.
                hatmu[t] = xt
                r = t
                alpha_r = alpha_r_from_restart(r, alpha)

        episode.submit(float(hatmu[t]))

    return episode.finish()
