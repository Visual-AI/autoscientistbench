"""Observation-only entry points for per-instance COT inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
from estimator import SmoothConditionalWasserstein, bootstrap_wavelet_cw


def algorithm_seed(arrays, seed):
    h = hashlib.sha256(str(seed).encode())
    for key in ('Y0', 'Z0', 'Y1', 'Z1'):
        x = np.ascontiguousarray(arrays[key], dtype='<f8')
        h.update(str(x.shape).encode())
        h.update(x.tobytes())
    return int.from_bytes(h.digest()[:8], 'little')


def estimate(arrays, config, seed):
    scw = SmoothConditionalWasserstein(dy=arrays['Y0'].shape[1], dz=arrays['Z0'].shape[1],
        J_joint=config['J_joint'], J_z=config['J_z'], wavelet=config['wavelet'],
        mode=config['mode'], threshold=config['threshold'], nonnegativity=True, renormalize=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Level value of .* is too high')
        result = bootstrap_wavelet_cw(scw=scw, **arrays, B=config['B'], alpha=config['alpha'],
                                     Nz=config['Nz'], Ny=config['Ny'], seed=seed)
    return result


def worker(input_path, index, config):
    started = time.monotonic()
    with np.load(input_path, allow_pickle=False) as z:
        iid = str(z['id'][index])
        arrays = {k: z[f'{k}_{index}'] for k in ('Y0', 'Z0', 'Y1', 'Z1')}
    result = estimate(arrays, config, algorithm_seed(arrays, config['seed']))
    return [iid, result.point_est, *result.ci_basic], time.monotonic()-started


def main(a):
    seed = int(os.environ.get('BENCH_SEED', '1'))
    cpus = max(1, int(os.environ.get('BENCH_CPUS', '1')))
    if a.command == 'train':
        # No cross-instance fitting in the source protocol.
        np.random.seed(seed)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        config = dict(B=100, alpha=.05, Nz=120, Ny=120, J_joint=4, J_z=6,
                      wavelet='db4', mode='periodization', threshold=None, seed=seed)
        (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        return
    config = json.loads((Path(a.model)/'config.json').read_text())
    if config['seed'] != seed:
        raise ValueError('BENCH_SEED must match the saved model configuration')
    with np.load(a.input, allow_pickle=False) as z:
        count = len(z['id'])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, 'w', newline='') as f, ProcessPoolExecutor(max_workers=cpus) as pool:
        writer=csv.writer(f); writer.writerow(['id','point','lower','upper']); f.flush()
        jobs=[pool.submit(worker, a.input, i, config) for i in range(count)]
        for j, future in enumerate(as_completed(jobs), 1):
            row, seconds = future.result(); writer.writerow(row); f.flush()
            print(f'completed {j}/{count}; instance_s={seconds:.3f}', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); sp=parser.add_subparsers(dest='command',required=True)
    t=sp.add_parser('train'); t.add_argument('--out',required=True); t.add_argument('--data'); t.add_argument('--seed')
    p=sp.add_parser('predict'); p.add_argument('--model',required=True); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    main(parser.parse_args())
