"""Command-line utilities for batch Bayesian optimization on a discrete condition table."""

import argparse
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge

def normalize_df(df):
    solvent_col = None
    for candidate in ("Sol.", "Solvent"):
        if candidate in df.columns:
            solvent_col = candidate
            break
    if solvent_col is not None:
        s = df[solvent_col].astype(str).str.replace('"', '', regex=False).str.strip()
        s_lower = s.str.lower()
        s = s.where(~s_lower.eq('tol'), 'Toluene')
        df[solvent_col] = s
    if 'Temp./℃' in df.columns:
        t = df['Temp./℃'].astype(str).str.extract(r'(\d+(?:\.\d+)?)', expand=False)
        df['Temp./℃'] = pd.to_numeric(t, errors='coerce')
    if 'equiv.' in df.columns:
        df['equiv.'] = df['equiv.'].astype(str).str.strip()
    if 'Yield' in df.columns:
        df['Yield'] = df['Yield'].astype(str).str.replace('%', '', regex=False)
        df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')

def build_preprocess(X):
    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category') or X[c].dtype == 'bool' or str(X[c].dtype) == 'str' or str(X[c].dtype).startswith('string')]
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocess = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    return preprocess

def normal_pdf(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * np.square(z))

def approx_cdf(z):
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * np.power(z, 3))))

def expected_improvement(mean, std, best, xi=0.0):
    # Standard EI used to score untested conditions.
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    imp = mean - best - xi
    with np.errstate(divide='ignore', invalid='ignore'):
        z = imp / std
        cdf = approx_cdf(z)
        pdf = normal_pdf(z)
        ei = imp * cdf + std * pdf
        ei = np.where(std <= 1e-12, 0.0, ei)
    return ei

def acquisition_scores(mean, std, best_train, acq='EI', xi=0.0, kappa=2.0):
    acq = (acq or 'EI').upper()
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if acq == 'EI':
        scores = expected_improvement(mean, std, best_train, xi=xi)
    elif acq == 'PI':
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (mean - best_train - xi) / std
            scores = approx_cdf(z)
            scores = np.where(std <= 1e-12, 0.0, scores)
    elif acq == 'UCB':
        scores = mean + kappa * std
    elif acq == 'LCB':
        scores = mean - kappa * std
    else:
        scores = expected_improvement(mean, std, best_train, xi=xi)
    return scores

def build_gp(n_features, random_state=None):
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-3))
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=random_state)

def fit_predict_surrogate(Xf, y, known, surrogate='linear', random_state=42):
    if str(surrogate).lower() == 'gp':
        gp = build_gp(Xf.shape[1], random_state=random_state)
        gp.fit(Xf[known], y.loc[known].values)
        mean, std = gp.predict(Xf, return_std=True)
        return mean, std
    br = BayesianRidge(compute_score=True)
    br.fit(Xf[known], y.loc[known].values)
    mu = br.coef_
    Sigma = br.sigma_
    intercept = br.intercept_
    noise_var = 1.0 / br.alpha_
    mean = Xf.dot(mu) + intercept
    var = np.einsum('ij,jk,ik->i', Xf, Sigma, Xf) + noise_var
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean, std

def propose_batch(df, target, init_indices=None, init_size=5, batch_size=5, acquisition='EI', xi=0.0, kappa=2.0, surrogate='linear', method='kriging', random_state=42):
    X = df.drop(columns=[target]).reset_index(drop=True)
    y = df[target].astype(float).reset_index(drop=True)
    N = len(y)
    rng = np.random.default_rng(random_state)
    known = np.zeros(N, dtype=bool)
    if init_indices is not None and len(init_indices) > 0:
        idx = [int(i) for i in init_indices if 0 <= int(i) < N]
        if len(idx) == 0:
            raise ValueError('init index out of range or empty')
        known[np.array(idx, dtype=int)] = True
    else:
        size = max(1, min(int(init_size), N))
        known[rng.choice(N, size=size, replace=False)] = True
    preprocess = build_preprocess(X)
    Xf = preprocess.fit_transform(X)
    if hasattr(Xf, 'toarray'):
        Xf = Xf.toarray()
    mean, std = fit_predict_surrogate(Xf, y, known, surrogate=surrogate, random_state=random_state)
    best_train = float(np.max(y.loc[known].values)) if np.any(known) else -np.inf
    scores = acquisition_scores(mean, std, best_train, acq=acquisition, xi=xi, kappa=kappa)
    scores = np.asarray(scores, dtype=float)
    unknown = ~known
    suggestions = []
    m = str(method).lower()
    if m == 'topk':
        s_idx = np.argsort(-scores)
        chosen = [int(i) for i in s_idx if unknown[i]][:int(batch_size)]
        for i in chosen:
            suggestions.append({'index': int(i), 'suggest_y': float(y.iloc[i]), 'pred_mean': float(mean[i]), 'pred_std': float(std[i]), 'acq': float(scores[i])})
    elif m == 'thompson':
        rng = np.random.default_rng(random_state)
        sample = mean + std * rng.standard_normal(size=N)
        s_idx = np.argsort(-sample)
        chosen = []
        for i in s_idx:
            if unknown[i]:
                chosen.append(int(i))
            if len(chosen) >= int(batch_size):
                break
        for i in chosen:
            suggestions.append({'index': int(i), 'suggest_y': float(y.iloc[i]), 'pred_mean': float(mean[i]), 'pred_std': float(std[i]), 'acq': float(scores[i])})
    else:
        chosen = []
        kb = np.where(known)[0].tolist()
        for _ in range(int(batch_size)):
            s_idx = np.argsort(-scores)
            next_i = None
            for i in s_idx:
                if (i not in chosen) and unknown[i]:
                    next_i = int(i)
                    break
            if next_i is None:
                break
            chosen.append(next_i)
            suggestions.append({'index': int(next_i), 'suggest_y': float(y.iloc[next_i]), 'pred_mean': float(mean[next_i]), 'pred_std': float(std[next_i]), 'acq': float(scores[next_i])})
            kb2 = kb + [next_i]
            if str(surrogate).lower() == 'gp':
                gp = build_gp(Xf.shape[1], random_state=random_state)
                y_believe = np.concatenate([y.iloc[kb].values, [mean[next_i]]])
                gp.fit(Xf[kb2], y_believe)
                mean, std = gp.predict(Xf, return_std=True)
            else:
                br = BayesianRidge(compute_score=True)
                y_believe = np.concatenate([y.iloc[kb].values, [mean[next_i]]])
                br.fit(Xf[kb2], y_believe)
                mu = br.coef_
                Sigma = br.sigma_
                intercept = br.intercept_
                noise_var = 1.0 / br.alpha_
                mean = Xf.dot(mu) + intercept
                var = np.einsum('ij,jk,ik->i', Xf, Sigma, Xf) + noise_var
                std = np.sqrt(np.maximum(var, 1e-12))
            best_train = float(np.max(np.concatenate([y.iloc[kb].values, [float(y.iloc[next_i])]]))) if len(kb) > 0 else float(y.iloc[next_i])
            scores = acquisition_scores(mean, std, best_train, acq=acquisition, xi=xi, kappa=kappa)
        suggestions = suggestions
    dataset_best_idx = int(np.argmax(y.values))
    found_any_best = any(s['index'] == dataset_best_idx for s in suggestions)
    return {
        'best_index': dataset_best_idx,
        'best_y': float(y.iloc[dataset_best_idx]),
        'init_count': int(np.sum(known)),
        'batch_size': int(batch_size),
        'found_any_best': bool(found_any_best),
        'suggestions': suggestions
    }

def simulate_batch(df, target, init_indices=None, init_size=5, batch_size=5, rounds=5, acquisition='EI', xi=0.0, kappa=2.0, surrogate='linear', method='kriging', random_state=42):
    X = df.drop(columns=[target]).reset_index(drop=True)
    y = df[target].astype(float).reset_index(drop=True)
    N = len(y)
    rng = np.random.default_rng(random_state)
    known = np.zeros(N, dtype=bool)
    if init_indices is not None and len(init_indices) > 0:
        idx = [int(i) for i in init_indices if 0 <= int(i) < N]
        if len(idx) == 0:
            raise ValueError('init index out of range or empty')
        known[np.array(idx, dtype=int)] = True
    else:
        size = max(1, min(int(init_size), N))
        known[rng.choice(N, size=size, replace=False)] = True
    preprocess = build_preprocess(X)
    Xf = preprocess.fit_transform(X)
    if hasattr(Xf, 'toarray'):
        Xf = Xf.toarray()
    dataset_best_idx = int(np.argmax(y.values))
    found_iter = None
    t = 0
    trace = []
    for r in range(1, int(rounds) + 1):
        mean, std = fit_predict_surrogate(Xf, y, known, surrogate=surrogate, random_state=random_state)
        best_train = float(np.max(y.loc[known].values)) if np.any(known) else -np.inf
        scores = acquisition_scores(mean, std, best_train, acq=acquisition, xi=xi, kappa=kappa)
        unknown = ~known
        chosen = []
        m = str(method).lower()
        if m == 'topk':
            s_idx = np.argsort(-scores)
            chosen = [int(i) for i in s_idx if unknown[i]][:int(batch_size)]
        elif m == 'thompson':
            rng = np.random.default_rng(random_state + r)
            sample = mean + std * rng.standard_normal(size=N)
            s_idx = np.argsort(-sample)
            for i in s_idx:
                if unknown[i]:
                    chosen.append(int(i))
                if len(chosen) >= int(batch_size):
                    break
        else:
            kb = np.where(known)[0].tolist()
            cur_mean = mean.copy()
            cur_std = std.copy()
            cur_scores = scores.copy()
            for _ in range(int(batch_size)):
                s_idx = np.argsort(-cur_scores)
                next_i = None
                for i in s_idx:
                    if (i not in chosen) and unknown[i]:
                        next_i = int(i)
                        break
                if next_i is None:
                    break
                chosen.append(next_i)
                kb2 = kb + [next_i]
                if str(surrogate).lower() == 'gp':
                    gp = build_gp(Xf.shape[1], random_state=random_state)
                    y_believe = np.concatenate([y.iloc[kb].values, [cur_mean[next_i]]])
                    gp.fit(Xf[kb2], y_believe)
                    cur_mean, cur_std = gp.predict(Xf, return_std=True)
                else:
                    br = BayesianRidge(compute_score=True)
                    y_believe = np.concatenate([y.iloc[kb].values, [cur_mean[next_i]]])
                    br.fit(Xf[kb2], y_believe)
                    mu = br.coef_
                    Sigma = br.sigma_
                    intercept = br.intercept_
                    noise_var = 1.0 / br.alpha_
                    cur_mean = Xf.dot(mu) + intercept
                    var = np.einsum('ij,jk,ik->i', Xf, Sigma, Xf) + noise_var
                    cur_std = np.sqrt(np.maximum(var, 1e-12))
                bt = float(np.max(np.concatenate([y.iloc[kb].values, [float(y.iloc[next_i])]]))) if len(kb) > 0 else float(y.iloc[next_i])
                cur_scores = acquisition_scores(cur_mean, cur_std, bt, acq=acquisition, xi=xi, kappa=kappa)
        for i in chosen:
            t += 1
            trace.append({'round': r, 'iter': t, 'suggest_index': int(i), 'suggest_y': float(y.iloc[i]), 'pred_mean': float(mean[i]), 'pred_std': float(std[i]), 'acq': float(scores[i]), 'current_best_y': float(np.max(y.loc[known].values)) if np.any(known) else -np.inf, 'found_best': bool(i == dataset_best_idx)})
            known[i] = True
            if i == dataset_best_idx and found_iter is None:
                found_iter = t
        if np.all(known):
            break
    init_idx_list = [int(i) for i in np.where(np.array([trace[0]['round'] == 1 if trace else False for _ in range(N)]) | (np.arange(N) < 0))[0]]
    if init_indices is not None and len(init_indices) > 0:
        init_idx_list = [int(i) for i in init_indices if 0 <= int(i) < N]
    else:
        rng2 = np.random.default_rng(random_state)
        init_idx_list = rng2.choice(N, size=max(1, min(int(init_size), N)), replace=False).tolist()
    init_trace = [{'round': 0, 'iter': idx + 1, 'suggest_index': int(i), 'suggest_y': float(y.iloc[i]), 'pred_mean': 0.0, 'pred_std': 0.0, 'acq': 0.0, 'current_best_y': 0.0, 'found_best': bool(i == dataset_best_idx)} for idx, i in enumerate(init_idx_list)]
    return {'best_index': dataset_best_idx, 'best_y': float(y.iloc[dataset_best_idx]), 'found_iter': None if found_iter is None else int(found_iter), 'final_known_count': int(np.sum(known)), 'init_indices': init_idx_list, 'trace': init_trace + trace}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--mode', type=str, default='batch')
    parser.add_argument('--init', type=str, default='')
    parser.add_argument('--init_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--acq', type=str, default='EI')
    parser.add_argument('--xi', type=float, default=0.0)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--surrogate', type=str, default='linear')
    parser.add_argument('--method', type=str, default='kriging')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--output', type=str, default='', help='Output CSV path. If left empty, results are printed to the terminal')
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    normalize_df(df)
    init_indices = [int(s) for s in str(args.init).split(',') if s.strip() != '']
    if str(args.mode).lower() == 'simulate_batch':
        res = simulate_batch(df, args.target, init_indices=init_indices if len(init_indices) > 0 else None, init_size=args.init_size, batch_size=args.batch_size, rounds=args.rounds, acquisition=args.acq, xi=args.xi, kappa=args.kappa, surrogate=args.surrogate, method=args.method, random_state=args.random_state)
        trace_df = pd.DataFrame(res['trace'])
        if args.output:
            trace_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
            print(f"best_index={res['best_index']}, best_y={res['best_y']}, found_iter={res['found_iter']}, final_known_count={res['final_known_count']}")
        else:
            summary = {'best_index': res['best_index'], 'best_y': res['best_y'], 'found_iter': res['found_iter'], 'final_known_count': res['final_known_count']}
            print(pd.Series(summary).to_string())
            print(trace_df.to_string(index=False))
    else:
        res = propose_batch(df, args.target, init_indices=init_indices if len(init_indices) > 0 else None, init_size=args.init_size, batch_size=args.batch_size, acquisition=args.acq, xi=args.xi, kappa=args.kappa, surrogate=args.surrogate, method=args.method, random_state=args.random_state)
        suggestions_df = pd.DataFrame(res['suggestions'])
        if args.output:
            suggestions_df.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
            print(f"best_index={res['best_index']}, best_y={res['best_y']}, init_count={res['init_count']}, batch_size={res['batch_size']}, found_any_best={res['found_any_best']}")
        else:
            summary = {'best_index': res['best_index'], 'best_y': res['best_y'], 'init_count': res['init_count'], 'batch_size': res['batch_size'], 'found_any_best': res['found_any_best']}
            print(pd.Series(summary).to_string())
            print(suggestions_df.to_string(index=False))

if __name__ == '__main__':
    main()
