"""Only Table 1 Bike cube-active; sequential R seed 0, no RNG changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(os.environ['PAPER_DIR'])
FEATURES = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
PARAMS = dict(n_estimators=1000, learning_rate=0.002, max_depth=7, random_state=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def train(out):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from models import ActiveInferenceModels
    data = ROOT / 'data/raw/hour.csv'
    df = pd.read_csv(data).dropna()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df['cnt'].to_numpy(dtype=float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    t0 = time.perf_counter()
    model = ActiveInferenceModels(dict(PARAMS), dict(PARAMS)).fit(Xtr, ytr)
    y_pred = model.predict_label(Xte)
    error_pred = model.error_model.predict(Xte)
    uncertainty = np.abs(error_pred)
    fit_s = time.perf_counter() - t0
    model.label_model.save_model(out / 'label_model.ubj')
    model.error_model.save_model(out / 'error_model.ubj')
    np.savez(out / 'test_predictions.npz', yte=yte, y_pred=y_pred,
             uncertainty=uncertainty, error_pred=error_pred)
    save_json(out / 'train.json', dict(xgb=PARAMS, features=FEATURES, split_seed=0,
        n_train=len(ytr), n_test=len(yte), theta_true=float(yte.mean()),
        prediction_rmse=float(np.sqrt(np.mean((yte-y_pred)**2))), fit_predict_s=fit_s,
        data_sha256=sha(data), predictions_sha256=sha(out / 'test_predictions.npz'),
        model_sha256={name:sha(out/name) for name in ['label_model.ubj', 'error_model.ubj']},
        threads=int(os.environ['OMP_NUM_THREADS'])))
    print(f'Trained both 1000-tree XGBoost models; fit/predict {fit_s:.3f}s', flush=True)


def cube_trials(d, n_trials=10000, variance_fn=None):
    from scipy.stats import norm
    from rpy2.robjects import r
    from sampling_methods import cube_active_sampling, compute_sampling_probabilities
    from variance_estimators import estimate_cube_variance
    if variance_fn is None:
        variance_fn = estimate_cube_variance
    y, pred, u, err = (d[k] for k in ['yte','y_pred','uncertainty','error_pred'])
    pi = compute_sampling_probabilities(u, 0.1, 0.5)
    auxiliary = err.reshape(-1, 1)
    residuals = y - pred
    theta = y.mean()
    z = norm.ppf(1.0 - (1.0 - 0.9) / 2.0)
    records = np.empty((n_trials, 5))
    r('set.seed(0)')
    for seed in range(n_trials):
        xi, pi_cube = cube_active_sampling(auxiliary, pi, random_state=seed)
        estimate = (pred + residuals * xi / pi_cube).mean()
        variance = variance_fn(residuals, auxiliary, xi, pi_cube, len(y))
        margin = z * np.sqrt(variance)
        lb, ub = estimate - margin, estimate + margin
        records[seed] = [(estimate-theta)**2, 2.0*margin, int(lb < theta < ub), lb, ub]
        if n_trials >= 1000 and (seed+1) % 1000 == 0:
            print(f'cube-active: {seed+1}/{n_trials}', flush=True)
    return records


def evaluate(out):
    meta = json.loads((out / 'train.json').read_text())
    assert meta['xgb'] == PARAMS
    assert sha(out/'test_predictions.npz') == meta['predictions_sha256']
    for name, digest in meta['model_sha256'].items():
        assert sha(out/name) == digest
    with np.load(out / 'test_predictions.npz') as d:
        trials = cube_trials(d)
    np.save(out / 'cube_trials.npy', trials)
    measured = dict(ci_width=float(trials[:,1].mean()), coverage=float(trials[:,2].mean()),
                    rmse=float(np.sqrt(trials[:,0].mean())))
    paper = dict(ci_width=0.3316, coverage=0.8969)
    previous = dict(ci_width=0.3288585630397503, coverage=0.8935)
    metrics = [dict(metric=k, measured=measured[k], paper=v,
                    difference=measured[k]-v, relative_difference=measured[k]/v-1,
                    previous_reproduction=previous[k], difference_previous=measured[k]-previous[k])
               for k,v in paper.items()]
    save_json(out / 'table.json', dict(target='Table 1 Bike cube-active', budget=0.1,
        confidence=0.9, n_trials=10000, r_seed=0, rng_changed=False, metrics=metrics,
        measured=measured, width_mc_se=float(trials[:,1].std(ddof=1)/np.sqrt(len(trials))),
        coverage_mc_se=float(np.sqrt(measured['coverage']*(1-measured['coverage'])/len(trials))),
        predictions_sha256=meta['predictions_sha256'], learning_rate=0.002))
    for row in metrics:
        print(f"{row['metric']}: measured={row['measured']:.8f}, paper={row['paper']:.4f}, difference={row['difference']:+.8f}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train','eval'])
    p.add_argument('out', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (train if a.stage == 'train' else evaluate)(a.out)
