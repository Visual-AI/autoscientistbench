"""File-based entry point for the unchanged SBI REINFORCE sampler."""
import argparse
import importlib.util
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from calibrated_guidance.guidance import build_reinforce_mean_fn
from calibrated_guidance.inference import flow_matching
from experiments.sbi.tasks import TASKS

REMAINING = mp.get_context('fork').Value('i', 1)


def worker(path):
    torch.set_num_threads(1)
    threadpool_limits(1)
    seed = int(os.environ.get('BENCH_SEED', '1')) - 1
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        task = TASKS[str(data['task'].item())]
        y = torch.from_numpy(data['observation'].copy())
    spec = importlib.util.spec_from_file_location('likelihood_env', Path(os.environ['BENCH_BENCH_DIR']) / 'likelihood_env.py')
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)
    likelihood = env.Likelihood(path.stem, task.name, y)
    posterior = task.build_posterior('cpu')
    mean_fn = build_reinforce_mean_fn(posterior, likelihood, num_samples_per_step=1000)
    time_steps = torch.linspace(1.0, 0.0, 100)
    if task.prior == 'uniform':
        time_steps[0] = 1.0 - 1e-3
    cpus = int(os.environ.get('BENCH_CPUS', '1'))

    def rethreaded_mean_fn(xt, t):
        # CPUs freed by finished instances go to the rest; draws are serial and kernels elementwise, so bytes are unchanged.
        torch.set_num_threads(max(1, cpus // max(1, REMAINING.value)))
        return mean_fn(xt, t)

    samples = flow_matching(rethreaded_mean_fn, time_steps, shape=(task.dim,), n_samples=10000, use_tqdm=False)
    with REMAINING.get_lock():
        REMAINING.value -= 1
    calls = likelihood.finish()
    print(f'Completed {path.stem}: {calls} likelihood evaluations', flush=True)
    return path.stem, samples.numpy(), calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    assert (Path(args.model) / 'NO_TRAINING').is_file()
    paths = sorted(Path(args.input).glob('*.npz'))
    assert paths
    torch.set_num_threads(1)
    threadpool_limits(1)
    # Independent complete instances; retain the authors' full per-instance RNG stream.
    workers = min(int(os.environ.get('BENCH_CPUS', '1')), len(paths))
    values = {}
    REMAINING.value = len(paths)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        for ident, samples, calls in pool.map(worker, paths):
            values[ident] = samples
            values[ident + '__calls'] = np.asarray(calls, dtype=np.int64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **values)
    tmp.replace(output)


if __name__ == '__main__':
    main()
