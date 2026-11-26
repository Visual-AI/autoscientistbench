"""File-based protocol for the unchanged per-isoform spatial test."""
import argparse
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import warnings

import numpy as np

DATA = None
CONFIG = None
SV = None


def load_method():
    # Load just the spatial module: plotting/preprocessing and package installation
    # are unnecessary for the simulation experiment.
    spec = importlib.util.spec_from_file_location(
        "spatial_method", Path(__file__).parent / "src/SplIsoFind/spatially_variable.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def instance_seed(coords, values, seed):
    # Independent algorithm RNG, based only on this observation, never its ID,
    # batch position, scenario identity, or data-generation seed.
    h = hashlib.blake2s(digest_size=4)
    h.update(str(seed).encode())
    h.update(np.asarray(coords, dtype='<f8').tobytes())
    h.update(np.asarray(values, dtype='i1').tobytes())
    return int.from_bytes(h.digest(), 'little')


def test_instance(coords, values, seed, config):
    import pandas as pd
    labels = pd.DataFrame(coords, columns=['x', 'y'])
    x = pd.Series(values, dtype=float)
    _, result = SV._compute_single_isoform(
        'synthetic_isoform', x, labels, ['All'], config['nperm'], config['k'],
        config['mincells'], config['mincellspergroup'], config['imb'],
        'x', 'y', seed, 'moransI')
    I, p = result.get('All', (float('nan'), float('nan')))
    called = bool(np.isfinite(I) and np.isfinite(p) and p <= config['alpha'] and I > config['I_threshold'])
    return int(called), float(I), float(p)


def worker_init():
    from threadpoolctl import threadpool_limits
    global LIMIT
    LIMIT = threadpool_limits(limits=1)
    warnings.filterwarnings('ignore', message='The weights matrix is not fully connected')


def compute(i):
    lo, hi = DATA['offsets'][i:i+2]
    coords, values = DATA['coords'][lo:hi], DATA['values'][lo:hi]
    seed = instance_seed(coords, values, CONFIG['seed'])
    called, _, _ = test_instance(coords, values, seed, CONFIG)
    return i, called


def main():
    global DATA, CONFIG, SV
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    train = sub.add_parser('train'); train.add_argument('--out', type=Path, required=True)
    pred = sub.add_parser('predict')
    pred.add_argument('--model', type=Path, required=True)
    pred.add_argument('--input', type=Path, required=True)
    pred.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    seed, cpus = int(os.environ.get('BENCH_SEED', '1')), int(os.environ.get('BENCH_CPUS', '1'))
    if cpus < 1:
        raise ValueError('BENCH_CPUS must be positive')
    if a.mode == 'train':
        a.out.mkdir(parents=True, exist_ok=True)
        cfg = dict(nperm=10000, k=50, mincells=250, mincellspergroup=20,
                   imb=.05, alpha=.05, I_threshold=.01, seed=seed)
        (a.out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
        return
    CONFIG = json.loads((a.model/'config.json').read_text())
    if CONFIG['seed'] != seed:
        raise ValueError('model and prediction BENCH_SEED disagree')
    with np.load(a.input, allow_pickle=False) as z:
        DATA = {k: z[k] for k in z.files}
    SV = load_method()
    n = len(DATA['id'])
    order = np.argsort(-np.diff(DATA['offsets']), kind='stable').tolist()
    results = [None]*n
    t0 = time.monotonic()
    with mp.get_context('fork').Pool(min(cpus, n), initializer=worker_init) as pool:
        for count, (i, called) in enumerate(pool.imap_unordered(compute, order, chunksize=1), 1):
            results[i] = called
            if count % 500 == 0 or count == n:
                print(f'{count}/{n} completed; {time.monotonic()-t0:.1f}s', flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'called'])
        w.writerows(zip(DATA['id'], results))


if __name__ == '__main__':
    main()
