import sys
import os

import numpy as np
import pandas as pd
import scipy as sp
from scipy.linalg import svd
from scipy.special import expit
from scipy.stats import norm

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from functools import reduce
from tqdm import tqdm

from PII import PII, fit_logistic, fit_rf


# Set the number of threads for various libraries
n_cores = len(os.sched_getaffinity(0))
n_cores = int(n_cores // 2 + 1)
os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_cores}"
os.environ["MKL_NUM_THREADS"] = f"{n_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_cores}"
os.environ["NUMBA_CACHE_DIR"] = '/tmp/numba_cache'

# Create results directory
path_result = 'results/ex1/'
os.makedirs(path_result, exist_ok=True)

# Simulation parameters
n_simu = 50
p = 1000
s = 0.2
d = 1
r = 10
rho = 1.0
sigma_U = 1.0
sigma_UE = 1.#sigma_UE = 0.8 # 

def compute_similarity_matrix(X):
    return np.dot(X, X.T)

def compute_projection_matrix_svd(A):
    """
    Computes the projection matrix P for the subspace defined by A using SVD.
    """
    U, _, _ = svd(A, full_matrices=False)
    P = np.dot(U, U.T)
    return P

def subspace_distance(A, B):
    """
    Calculates the distance between two subspaces using the principal angles.
    """
    S_A = compute_projection_matrix_svd(A)
    S_B = compute_projection_matrix_svd(B)
    distance = sp.linalg.norm(S_A - S_B, 2)
    return distance

def fit_glm_with_test(Y_C, Y, X, CC, B, U=None, link='logit', **kwargs):
    """
    Fits a generalized linear model (GLM) and performs hypothesis testing.
    """
    n, p = Y.shape

    if U is not None:
        X = np.hstack((X, U))
    if link == 'logit':
        reg = sm.Logit
    elif link == 'identity':
        reg = sm.OLS

    conf = []
    p_values = np.zeros(len(CC))
    for i, ic in enumerate(CC):
        model = reg(Y[:, ic], X)
        result = model.fit(disp=False)
        conf.append(result.conf_int()[0, :])
        p_values[i] = result.pvalues[0]

    conf = np.array(conf)
    is_covered = (conf[:, 0] < B[:, CC]) & (B[:, CC] < conf[:, 1])
    sig = p_values < 0.05
    return p_values, is_covered, sig[None, :], U

def fit_semi(Y_C, Y, X, CC, B, U=None, r=r):
    """
    Fits a semi-parametric model using PII and performs hypothesis testing.
    """
    if U is None:
        U = lambda Y, Y_test: PCA(n_components=r).fit_transform(Y_C)

    X_U = fit_rf
    Y_XU = fit_rf

    res = PII(Y, X, [], U, X_U, Y_XU, Y_C=Y_C, link='logit', crossfit=1)
    p_values = res['p_values']
    hbeta = res['hbeta']
    var_beta = res['var_beta']
    U_hat = res['U_hat']

    z_score = norm.ppf(1 - 0.05 / 2)
    lower = hbeta - z_score * np.sqrt(var_beta)
    upper = hbeta + z_score * np.sqrt(var_beta)

    is_covered = (lower < B[:, CC]) & (B[:, CC] < upper)
    sig = p_values < 0.05
    return p_values[0, :], is_covered, sig, U_hat

def generate_data(m, n, p, s, seed, ratio_nonnull=0., rho=rho, sigma_U=sigma_U, sigma_UE=sigma_UE):
    """
    Generates synthetic data for the simulation.
    """
    np.random.seed(seed)
    X = np.random.normal(size=(m + n, d))
    B = rho * np.random.binomial(1, s, size=(1, int(p*3/2)))

    U = X @ np.random.uniform(size=(d, r), low=-1, high=1) + sigma_UE * np.random.normal(size=(m + n, r))
    BU = np.random.uniform(size=(r, int(p*3/2)), low=-1, high=1) / np.sqrt(r)
    func = lambda u: sigma_U * (BU.T @ u)

    Theta = X @ B + np.apply_along_axis(func, 1, U)
    prob = expit(Theta)
    Y = np.random.binomial(1, prob)
    
    if ratio_nonnull is not None:
        C = np.random.choice(np.where(B[0, :p] == 0)[0], int(p / 2), replace=False)
        n_mis = int(len(C) * ratio_nonnull)
        C[:n_mis] = np.random.choice(np.where(B[0, :p] != 0)[0], n_mis, replace=False)
    elif ratio_nonnull is None:
        p_values, is_covered, sig, U_hat = fit_glm_with_test(None, Y[:,:p], X, np.arange(p), B[:,:p], U=None)
#         print(np.sort(-p_values)[:10], B[0,np.argsort(-p_values)[:10]])
#         C = np.argsort(-p_values)[:int(p / 2)]
        C = np.random.choice(np.where(B[0, :p] == 0)[0], int(p / 2), replace=False)
    return Y[:, C], Y[:n,p:], X[:n], U[:n], C, B[:,p:]

def run_simu(m, n, p, s, seed, ratio_nonnull, r=r):
    """
    Runs a single simulation.
    """
    Y_C, Y, X, U, C, B = generate_data(m, n, p, s, seed, ratio_nonnull, rho=rho, sigma_U=sigma_U, sigma_UE=sigma_UE)
    CC = np.arange(int(p/2))

    res = []
    for func, func_name in zip([fit_glm_with_test, fit_semi], ['GLM', 'PII']):
        for _U, has_U in zip([None, U], [False, True]):
            p_values, is_covered, sig, U_hat = func(Y_C, Y, X, CC, B, U=_U)
            coverage = np.mean(is_covered)
            power = np.sum(sig & (B[:, CC] != 0)) / np.sum(B[:, CC] != 0)
            fdr = np.sum(sig & (B[:, CC] == 0)) / np.sum(sig)
            type1_err = np.sum(sig & (B[:, CC] == 0)) / np.sum(B[:, CC] == 0)
            num_dis = np.sum(sig)

            # Apply BH procedure
            _, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            sig_bh = pvals_corrected < 0.05
            fdr_bh = np.sum(sig_bh & (B[0, CC] == 0)) / np.sum(sig_bh)

            if not has_U and func_name == 'PII':
                err_U = subspace_distance(U, U_hat)
            else:
                err_U = np.nan

            res.append([func_name, has_U, n, p, s, coverage, power, fdr, fdr_bh, type1_err, num_dis, err_U])

    return res

# Run simulations in parallel
n = 300
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
    m = 0
    for ratio_nonnull in [0., 0.05, 0.1, 0.2]:
        df_res = pd.DataFrame()
        res = parallel(
            delayed(run_simu)(m, n, p, s, seed, ratio_nonnull)
            for seed in tqdm(np.arange(n_simu), desc='seed')
        )

        res = pd.DataFrame(np.concatenate(res, axis=0), columns=[
            'method', 'has_U', 'n', 'p', 's', 'coverage', 'power', 'fdr', 'fdr_BH', 'type1_err', 'num_dis', 'err_U'
        ])
        df_res = pd.concat([df_res, res], axis=0)

        df_res.to_csv('{}res_{}_{}_sigma_{:.01f}_ratio_nonnull_{:.02f}.csv'.format(
            path_result, n, p, sigma_UE, ratio_nonnull), index=False)



    df_res = pd.DataFrame()
    res = parallel(
        delayed(run_simu)(m, n, p, s, seed, None)
        for seed in tqdm(np.arange(n_simu), desc='seed')
    )

    res = pd.DataFrame(np.concatenate(res, axis=0), columns=[
        'method', 'has_U', 'n', 'p', 's', 'coverage', 'power', 'fdr', 'fdr_BH', 'type1_err', 'num_dis', 'err_U'
    ])
    df_res = pd.concat([df_res, res], axis=0)

    df_res.to_csv('{}res_{}_{}_sigma_{:.01f}_least_var.csv'.format(
        path_result, n, p, sigma_UE), index=False)
