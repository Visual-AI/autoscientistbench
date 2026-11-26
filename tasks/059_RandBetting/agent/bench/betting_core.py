"""Verbatim used functions from confseq 0.0.11 (MIT License).
Unused imports and unrelated functions omitted for portable NumPy-only packaging.
"""
import math
import numpy as np

def lambda_predmix_eb(
    x,
    truncation=math.inf,
    alpha=0.05,
    fixed_n=None,
    prior_mean=1 / 2,
    prior_variance=1 / 4,
    fake_obs=1,
    scale=1,
):
    """
    Predictable mixture lambda values or "bets"

    Parameters
    ----------
    x, array-like of 0-1 bounded reals
        Observed data

    truncation, positive real or infinity
        Level at which to truncate lambda

    alpha, (0, 1)-valued real
        Significance level in (0, 1)

    fixed_n, positive integer or None
        Sample size for which lambda should be optimized.
        If left as None, lambda will scale like 1/sqrt{t log t}

    prior_mean, [0, 1]-valued real
        Prior mean to use for regularized sample mean

    prior_variance, (0, 1/4]-valued real
        Prior variance to use for regularized sample variance

    fake_obs, positive integer
        Number of 'fake observations' to add.
        Larger values correspond to more regularization near
        `prior_mean` and `prior_variance`

    scale, positive real
        Scale by which to multiply final lambda output.
        For most applications, this should be left as 1

    Returns
    -------
    lambdas, array-like of positive reals
        A (numpy) array of lambda values or "bets"
    """
    t = np.arange(1, len(x) + 1)
    mu_hat_t = np.minimum((fake_obs * prior_mean + np.cumsum(x)) / (t + fake_obs), 1)
    mu_hat_tminus1 = np.append(prior_mean, mu_hat_t[0 : (len(x) - 1)])
    sigma2_t = (fake_obs * prior_variance + np.cumsum(np.power(x - mu_hat_t, 2))) / (
        t + fake_obs
    )
    sigma2_tminus1 = np.append(prior_variance, sigma2_t[0 : (len(x) - 1)])
    if fixed_n is None:
        lambdas = np.sqrt(2 * np.log(1 / alpha) / (t * np.log(1 + t) * sigma2_tminus1))
    else:
        lambdas = np.sqrt(2 * np.log(1 / alpha) / (fixed_n * sigma2_tminus1))

    lambdas[np.isnan(lambdas)] = 0

    lambdas = np.minimum(truncation, lambdas)

    return lambdas * scale

def lambda_LBOW(x, m):
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    mu_hat_t = (1 / 2 + S_t) / (t + 1)
    mu_hat_tminus1 = np.append(1 / 2, mu_hat_t[0 : (len(mu_hat_t) - 1)])
    sigma2_tminus1 = np.append(1 / 4, np.cumsum(np.power(x - m, 2)))[0 : len(x)] / (
        t + 1
    )

    g = mu_hat_tminus1 - m
    M = np.where(g > 0, m, 1 - m)
    lambdas = g / (M * np.abs(g) + sigma2_tminus1 + np.power(g, 2))
    return lambdas

def betting_mart(
    x,
    m,
    alpha=0.05,
    lambdas_fn_positive=None,
    lambdas_fn_negative=None,
    N=None,
    convex_comb=False,
    theta=1 / 2,
    trunc_scale=1 / 2,
    m_trunc=True,
):
    """
    Betting martingale for a given sequence
    of data and a null mean value.

    Parameters
    ----------
    x, array-like
        The vector of observations.

    m, real
        Null value for the mean of x

    alpha, real
        Significance level between 0 and 1.

    lambdas_fn_postive, bivariate function or None
        Function of `x` and `m` which generates an array-like
        of bets with the same length as `x`.

    lambdas_fn_negative=None, bivariate function or None
        Same as above but for the negative capital process.
        This will be set to lambdas_fn_positive if
        left as None.

    N, positive integer or None
        Population size if sampling WoR

    convex_comb, boolean
        Should a convex combination of martingales be taken?
        If True, the process will be theta*pcp + (1-theta)*ncp,
        and if False, it will be max(theta*pcp, (1-theta)*ncp) where
        pcp and ncp are the positive and negative capital processes,
        respectively.

    theta, [0, 1]-valued real
        Positive/negative capital process weight as
        defined in `convex_comb`

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation defined
        in `m_trunc`. Leaving this as 1 will perform no
        additional truncation.

    m_trunc, boolean
        Should truncation be used based on m? If True, then truncation
        will be given by trunc_scale * 1/m or trunc_scale * 1/(1-m)
        depending on the capital process. If False, then truncation
        will be given by trunc_scale.


    Returns
    -------
    mart, array-like
        The martingale that results from the observed x
    """

    assert 0 <= theta <= 1

    # if theta == 1:
    #     assert all(x >= 0)
    # elif theta == 0:
    #     assert all(x <= 1)
    # else:
    #     assert all(x >= 0) and all(x <= 1)

    if lambdas_fn_positive is None:
        lambdas_fn_positive = lambda x, m: lambda_predmix_eb(x, alpha=alpha * theta)

    if lambdas_fn_negative is None:
        lambdas_fn_negative = lambda x, m: lambda_predmix_eb(
            x, alpha=alpha * (1 - theta)
        )

    if N is not None:
        t = np.arange(1, len(x) + 1)
        S_t = np.cumsum(x)
        S_tminus1 = np.append(0, S_t[0 : (len(x) - 1)])
        mu_t = (N * m - S_tminus1) / (N - (t - 1))
    else:
        mu_t = np.repeat(m, len(x))

    assert 0 < trunc_scale <= 1

    if theta > 0:
        lambdas_positive = lambdas_fn_positive(x, m)
        # if we want to truncate with m
        if m_trunc:
            with np.errstate(divide="ignore"):
                lambdas_positive = np.minimum(lambdas_positive, trunc_scale / mu_t)
                lambdas_positive = np.maximum(
                    lambdas_positive, -trunc_scale / (1 - mu_t)
                )
        else:
            lambdas_positive = np.minimum(lambdas_positive, trunc_scale)
            lambdas_positive = np.maximum(lambdas_positive, -trunc_scale)

        with np.errstate(invalid="ignore"):
            multiplicand_positive = 1 + lambdas_positive * (x - mu_t)

        # Use convention that inf * 0 = 0. We still have
        # a martingale under the null
        multiplicand_positive[
            np.logical_and(lambdas_positive == math.inf, x - mu_t == 0)
        ] = 1

        with np.errstate(invalid="ignore"):
            capital_process_positive = np.cumprod(multiplicand_positive)

        # If we get nans from 0 * inf, this should be 0
        capital_process_positive[np.isnan(capital_process_positive)] = 0
    else:
        capital_process_positive = np.repeat(0, len(x))

    if theta < 1:
        lambdas_negative = lambdas_fn_negative(x, m)
        if m_trunc:
            with np.errstate(divide="ignore"):
                lambdas_negative = np.minimum(
                    lambdas_negative, trunc_scale / (1 - mu_t)
                )
                lambdas_negative = np.maximum(lambdas_negative, -trunc_scale / mu_t)
        else:
            lambdas_negative = np.minimum(lambdas_negative, trunc_scale)
            lambdas_negative = np.maximum(lambdas_negative, -trunc_scale)

        with np.errstate(invalid="ignore"):
            multiplicand_negative = 1 - lambdas_negative * (x - mu_t)

        multiplicand_negative[
            np.logical_and(lambdas_negative == math.inf, x - mu_t == 0)
        ] = 1

        with np.errstate(invalid="ignore"):
            capital_process_negative = np.cumprod(multiplicand_negative)

        capital_process_negative[np.isnan(capital_process_negative)] = 0
    else:
        capital_process_negative = np.repeat(0, len(x))

    if theta == 1:
        capital_process = capital_process_positive
    elif theta == 0:
        capital_process = capital_process_negative
    else:
        if convex_comb:
            capital_process = (
                theta * capital_process_positive
                + (1 - theta) * capital_process_negative
            )
        else:
            capital_process = np.maximum(
                theta * capital_process_positive, (1 - theta) * capital_process_negative
            )

    # If mu_t < 0 or mu_t > 1, we cannot be under the null
    capital_process[np.logical_or(mu_t < 0, mu_t > 1)] = math.inf

    assert not any(np.isnan(capital_process))

    assert all(capital_process >= 0)

    return capital_process

