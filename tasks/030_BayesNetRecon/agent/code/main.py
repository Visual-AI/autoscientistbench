#!/usr/bin/env python3
"""Protocol adapter for the published observed-update BERM rolling calculation."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import numpy as np
from or4cle import BERM_solver


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def week(label):
    y, w = label.split('-W')
    return int(y), int(w)


def train(args):
    data = Path(args.data)
    rows = read(data / 'train/weekly.csv')
    point, jack = {}, {}
    for r in read(data / 'train/pstar.csv'):
        lab = r['label'].split('_jk')[0]
        (jack if r['kind'] == 'jackknife' else point).setdefault(lab, []).append(float(r['p_star']))
    for r in rows:
        point.setdefault(r['id'], [float(r['L']) / float(r['V'])])
    # Initial calibration uses the reproduction's points-then-jackknife order.
    pts = [p for r in rows for p in point[r['id']]]
    jks = [p for r in rows for p in jack.get(r['id'], [])]
    prior = BERM_solver.Prior.beta_from_samples(pts + jks)
    model = {'point': point, 'jack': jack, 'initial_prior': prior.params,
             'cal_window_years': 3, 'mode': 'gh', 'gh_nodes': 25,
             'gh_max_iter': 60, 'gh_tol': 1e-10, 'seed': args.seed}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.json').write_text(json.dumps(model, indent=2)+'\n')
    print(json.dumps({'initial_prior': prior.params, 'seed_weeks': len(rows)}))


def predict(args):
    model = json.loads((Path(args.model) / 'model.json').read_text())
    rows = read(args.input)
    point, jack = model['point'], model['jack']
    if [r['id'] for r in rows] != sorted({r['id'] for r in rows}):
        raise ValueError('history must be chronological with unique weeks')
    rng = np.random.default_rng(args.seed)
    predictions = []
    for i, target in enumerate(rows):
        if target['forecast'] != '1':
            continue
        if i == 0:
            raise ValueError('one-step prediction requires observed history')
        current = rows[i-1]
        y, w = week(current['id'])
        lim = (y-model['cal_window_years'], w)
        samples = []
        # Inclusive ISO-year/week bounds, and the same within-week sample order
        # as the reproduction. Only strictly earlier rows enter the window.
        for r in rows[:i]:
            if lim <= week(r['id']) <= (y, w):
                samples.extend(point.get(r['id'], [float(r['L']) / float(r['V'])]))
                samples.extend(jack.get(r['id'], []))
        prior = BERM_solver.Prior.beta_from_samples(samples)
        p_pred, _, _ = BERM_solver.berm_predictive_probabilities(
            int(current['N']), float(current['L']), prior,
            mode=model['mode'], return_Q=False, gh_nodes=model['gh_nodes'],
            gh_max_iter=model['gh_max_iter'], gh_tol=model['gh_tol'], rng=rng)
        predictions.append((target['id'], float(target['V']) * float(p_pred)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(predictions)
    print('wrote %d predictions' % len(predictions))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--out', required=True)
    t.add_argument('--data', required=True)
    t.add_argument('--seed', type=int, default=1)
    p = sub.add_parser('predict')
    for name in ('model', 'input', 'output'):
        p.add_argument('--'+name, required=True)
    args = ap.parse_args()
    args.seed = int(os.environ.get('BENCH_SEED', getattr(args, 'seed', 1)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    train(args) if args.cmd == 'train' else predict(args)
