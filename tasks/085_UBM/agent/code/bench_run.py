"""Protocol adapter: unchanged logistic fits, per observed experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from synthetic.utils_estimate import fit_models, pearsonr
from synthetic.utils_data_gen import make_preds, merge_df_val

# Load the frozen schema by absolute path; editable code cannot shadow it.
spec = importlib.util.spec_from_file_location('ubm_format', Path(os.environ['BENCH_BENCH_DIR'])/'format.py')
fmt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmt)
PREDICTORS = [f'Xp{i+1}' for i in range(32)]


def dataframe(observed, r):
    """Reconstruct author features; unobserved outcomes have unused -1 sentinels."""
    x = observed[:, :5].astype(np.int64)
    s, a, y = observed[:, 5:8].astype(np.int64).T
    group = x @ (2**np.arange(4, -1, -1))
    data = dict(R=np.full(len(x), r, dtype=np.int64))
    data.update({f'X{i+1}': x[:, i] for i in range(5)})
    data.update(S=s, A=a, Y=y, Y0=np.where(a == 0, y, -1), Y1=np.where(a == 1, y, -1))
    data.update({key: (group == i).astype(np.int64) for i, key in enumerate(PREDICTORS)})
    return pd.DataFrame(data)


def fit_observed(observed):
    start = time.monotonic()
    rct = dataframe(observed[:50000], 1)
    obs = dataframe(observed[50000:100000], 0)
    rct_models = fit_models(rct, PREDICTORS)
    obs_models = fit_models(obs, PREDICTORS)
    rv = dataframe(observed[100000:140000], 1)
    ov = dataframe(observed[140000:], 0)
    make_preds(rv, PREDICTORS, rct_models)
    make_preds(ov, PREDICTORS, obs_models)
    df = merge_df_val(rv, ov, PREDICTORS, rct_models, obs_models)
    stats = [v for key in fmt.SIGNALS for v in pearsonr(df, 'abs(b1(X))', f'SE_{key}', 2000)]
    support = pd.DataFrame(np.eye(32, dtype=np.int64), columns=PREDICTORS)
    probs = np.concatenate([models[key].predict_proba(support)[:, -1]
                            for models in (rct_models, obs_models) for key in fmt.MODEL_KEYS])
    return np.r_[probs, stats], time.monotonic()-start


def work(job):
    path, iid, seed = job
    np.random.seed(seed)
    with np.load(path, allow_pickle=False) as z:
        observed = z['o_'+iid]
    values, elapsed = fit_observed(observed)
    return iid, values, elapsed


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='command', required=True)
    t = sp.add_parser('train'); t.add_argument('--out', required=True)
    q = sp.add_parser('predict')
    for arg in ('model', 'input', 'output'): q.add_argument('--'+arg, required=True)
    a = p.parse_args()
    seed = int(os.environ['BENCH_SEED']); cpus = max(1, int(os.environ['BENCH_CPUS']))
    if a.command == 'train':
        out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
        (out/'config.json').write_text(json.dumps({'method': 'author_LogisticRegression_defaults',
                                                  'n_val': 2000, 'd_measured': 5, 'seed': seed})+'\n')
        print('Per-instance fitting takes place in predict; configuration saved.')
        return
    cfg = json.loads((Path(a.model)/'config.json').read_text())
    assert cfg['n_val'] == 2000 and cfg['d_measured'] == 5
    path = str(Path(a.input).resolve())
    with np.load(path, allow_pickle=False) as z: ids = z['id'].astype(str).tolist()
    with open(a.output, 'w', newline='') as f, ProcessPoolExecutor(max_workers=min(cpus, len(ids))) as pool:
        writer = csv.writer(f); writer.writerow(fmt.COLS)
        for k, (iid, values, elapsed) in enumerate(pool.map(work, [(path, i, seed) for i in ids])):
            writer.writerow([iid]+values.tolist())
            if (k+1) % 20 == 0 or k+1 == len(ids):
                print(f'Completed {k+1}/{len(ids)}; last instance {elapsed:.3f}s', flush=True)

if __name__ == '__main__': main()
