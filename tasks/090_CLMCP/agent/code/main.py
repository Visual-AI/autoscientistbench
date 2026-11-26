"""Benchmark entry points: released exact ATC with a streaming data adapter."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from icml.atc_scaling_scenarios_dense_adver import atc_run, candidate_splits_all, gamma_paper


def load_env():
    path = Path(os.environ['BENCH_BENCH_DIR']) / 'stream_env.py'
    spec = importlib.util.spec_from_file_location('frozen_stream_env', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_episode(group):
    env = load_env()
    episode = env.make_episode(group['seed'], group['x'])
    record = atc_run(episode, sigma=1.0, alpha=0.05,
                     split_fn=candidate_splits_all, gamma_fn=gamma_paper)
    return [(i, p) for i, p in zip(group['ids'], record['estimates'])]


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='command', required=True)
    t = sp.add_parser('train'); t.add_argument('--out', required=True)
    t.add_argument('--data'); t.add_argument('--seed')
    p = sp.add_parser('predict'); p.add_argument('--model', required=True)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    a = ap.parse_args()
    seed = int(os.environ.get('BENCH_SEED', '1'))
    rng = np.random.default_rng(seed)  # algorithm stream, unused by deterministic ATC
    if a.command == 'train':
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / 'model.json').write_text(json.dumps({'method':'ATC-exact','sigma':1.0,'alpha':0.05,'seed':seed}))
        return
    json.loads((Path(a.model) / 'model.json').read_text())
    groups = load_env().read_input(a.input)
    workers = min(int(os.environ.get('BENCH_CPUS', '1')), len(groups))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(run_episode, groups))
    else:
        records = [run_episode(g) for g in groups]
    with open(a.output, 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['id','pred'])
        for rows in records:
            writer.writerows(rows)

if __name__ == '__main__':
    main()
