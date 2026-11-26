"""Benchmark entry points; one independent process per online episode."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path


def load_env():
    path = Path(os.environ['BENCH_BENCH_DIR']) / 'clasp_env.py'
    spec = importlib.util.spec_from_file_location('frozen_clasp_env', path)
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)
    return env


def trial(row):
    import numpy as np
    from clasp import CLASP_I, T, F, G, D
    key, instance_seed = row
    env = load_env()
    # Available for stochastic candidate methods. The baseline is deterministic
    # conditional on the public, environment-provided initial point.
    algorithm_rng = np.random.default_rng(int(os.environ.get('BENCH_SEED', '1')))
    ep = env.make_episode(instance_seed)
    algorithm = CLASP_I(ep.initial(), T, (F, G, D))
    final_x, _ = algorithm.run(ep)
    return key, ep.finish(final_x)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('train'); q.add_argument('--out', required=True); q.add_argument('--data'); q.add_argument('--seed', type=int, default=1)
    q = sub.add_parser('predict'); q.add_argument('--model', required=True); q.add_argument('--input', required=True); q.add_argument('--output', required=True)
    a = p.parse_args()
    if a.command == 'train':
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out/'model.json').write_text(json.dumps({'method':'CLASP-I','seed':a.seed,'offline_training':False})+'\n')
        return
    json.loads((Path(a.model)/'model.json').read_text())
    env = load_env(); rows = env.read_input(a.input)
    workers = min(max(1, int(os.environ.get('BENCH_CPUS', '1'))), len(rows))
    if workers == 1:
        result = list(map(trial, rows))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
            result = list(pool.map(trial, rows, chunksize=1))
    env.write_output(a.output, result)


if __name__ == '__main__':
    main()
