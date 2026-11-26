"""Offline benchmark entry points; numerical Tera implementation is upstream."""
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

DATASETS = ['DHA', 'AT-AT', 'stachyose', 'AT-AT-CG-CG', 'buckyball-catcher', 'double-walled-nanotube']


def worker(a):
    import numpy as np
    import torch
    from tera.methods.tera.model import tera_model
    from tera.methods.common.training import regression_data
    from tera.methods.common.random import seed_for, random_stream
    torch.set_num_threads(min(4, int(os.environ.get('BENCH_CPUS', '64'))))
    torch.set_num_interop_threads(1)
    out = Path(a.model) / f'd{a.dataset}_r{a.replica}.pt'
    if a.stage == 'fit':
        with np.load(Path(os.environ['BENCH_DATA_DIR']) / 'train' / f'd{a.dataset}.npz') as z:
            r, e, f = (torch.from_numpy(z[k].copy()) for k in ['R', 'E', 'F'])
        # Materialized rows are sorted. Restore the paper's random input ordering
        # using the run's public algorithm seed; this never re-splits any data.
        seed = int(os.environ.get('BENCH_SEED', '1')) * 3 + a.replica
        order = torch.randperm(len(e), generator=torch.Generator().manual_seed(seed))
        r, e, f = r[order], e[order], f[order]
        mean, std = e.mean(), e.std().clamp_min(e.new_tensor(1e-12))
        split = regression_data(x_train=(r.reshape(len(r), -1) / 3.).contiguous(),
                                y_train=((-e + mean) / std).contiguous(),
                                g_train=(3. * f.reshape(len(f), -1) / std).contiguous())
        model = tera_model(m=20, kernel='rbf', outputscale=1., sigma_f=.001, sigma_g=.001,
                           lengthscale=1., lengthscale_init='median', lengthscale_init_max_points=2048,
                           use_ard=False, seed=seed_for(seed, random_stream.training),
                           train_steps=0, train_epochs=1, graph_refresh_epochs=0,
                           train_batch_size=256, prediction_batch_size=32, lr=.01, weight_decay=0.,
                           learn_lengthscale=True, learn_outputscale=True, learn_sigma_f=True,
                           learn_sigma_g=True, min_sigma_f=1e-6, min_sigma_g=0., log_every=25,
                           gradient_noise_model=None)
        model.fit(split)
        torch.save({'model': model, 'energy_mean': mean, 'energy_std': std, 'seed': seed}, out)
    else:
        artifact = torch.load(out, map_location='cpu', weights_only=False)
        key = f'd{a.dataset}_r{a.replica}'
        with np.load(a.input) as z:
            ids = z[key + '_id']; r = torch.from_numpy(z[key + '_R'].copy())
        with torch.no_grad():
            p = artifact['model'].predict(r.reshape(len(r), -1) / 3.).y_mean.reshape(-1)
            energy = artifact['energy_mean'] - artifact['energy_std'] * p
        with open(Path(a.output) / (key + '.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerows(zip(ids, energy.tolist()))


def dispatch(a):
    cpus = int(os.environ.get('BENCH_CPUS', '64'))
    workers = max(1, min(12, cpus // 4))
    out = Path(a.out if a.command == 'train' else a.model)
    out.mkdir(parents=True, exist_ok=True)
    if a.command == 'train' and list(out.glob('*.pt')):
        raise RuntimeError('Training requires a fresh output directory')
    if a.command == 'predict':
        import tempfile
        temp = tempfile.TemporaryDirectory(prefix='tera-pred-')
    def run(job):
        d, r = job
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
               '--stage', 'fit' if a.command == 'train' else 'infer',
               '--model', str(out), '--dataset', str(d), '--replica', str(r)]
        if a.command == 'predict':
            cmd += ['--input', a.input, '--output', temp.name]
        env = os.environ.copy()
        for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            env[key] = str(min(4, cpus))
        print('start', a.command, DATASETS[d], r, flush=True)
        subprocess.run(cmd, env=env, check=True)
        print('finished', a.command, DATASETS[d], r, flush=True)
    jobs = [(d, r) for d in range(6) for r in range(3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, jobs))
    if a.command == 'predict':
        # Resource columns are appended by the frozen formatter after monitoring ends.
        with open(a.output, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'pred'])
            for d, r in jobs:
                with open(Path(temp.name) / f'd{d}_r{r}.csv', newline='') as q:
                    w.writerows(csv.reader(q))
        temp.cleanup()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--out', required=True)
    e = sub.add_parser('predict')
    for k in ['model', 'input', 'output']: e.add_argument('--'+k, required=True)
    w = sub.add_parser('worker'); w.add_argument('--stage', required=True)
    for k in ['dataset', 'replica']: w.add_argument('--'+k, type=int, required=True)
    for k in ['model', 'input', 'output']: w.add_argument('--'+k)
    a = p.parse_args()
    worker(a) if a.command == 'worker' else dispatch(a)
