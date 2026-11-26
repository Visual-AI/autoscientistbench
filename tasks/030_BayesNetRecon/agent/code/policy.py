"""One-step BERM policy. The trusted scorer releases outcomes after prediction."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from or4cle import BERM_solver


def week(label):
    year, number = label.split('-W')
    return int(year), int(number)


class Policy:
    def __init__(self, model_dir):
        self.model = json.loads((Path(model_dir) / 'model.json').read_text())
        self.rows = []
        self.rng = np.random.default_rng(int(os.environ.get('BENCH_SEED', '1')))

    def handle(self, message):
        if message['op'] == 'observe':
            row = message['row']
            if self.rows and week(row['id']) <= week(self.rows[-1]['id']):
                raise ValueError('observations must be chronological')
            self.rows.append(row)
            return {'observed': row['id']}
        if message['op'] != 'forecast':
            raise ValueError('unknown operation')
        target = message['target']
        if not self.rows:
            raise ValueError('one-step forecast needs observed history')
        current = self.rows[-1]
        y, w = week(current['id'])
        lower = (y - self.model['cal_window_years'], w)
        samples = []
        for row in self.rows:
            if lower <= week(row['id']) <= (y, w):
                samples.extend(self.model['point'].get(
                    row['id'], [float(row['L']) / float(row['V'])]))
                samples.extend(self.model['jack'].get(row['id'], []))
        prior = BERM_solver.Prior.beta_from_samples(samples)
        p_pred, _, _ = BERM_solver.berm_predictive_probabilities(
            int(current['N']), float(current['L']), prior,
            mode=self.model['mode'], return_Q=False,
            gh_nodes=self.model['gh_nodes'],
            gh_max_iter=self.model['gh_max_iter'],
            gh_tol=self.model['gh_tol'], rng=self.rng)
        return {'id': target['id'], 'alpha': float(prior.params['alpha']),
                'beta': float(prior.params['beta']), 'p': float(p_pred)}


def main():
    from private_rpc import call
    parser = argparse.ArgumentParser()
    for name in ('model', 'input', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    policy = Policy(args.model)
    while True:
        event = call('next')
        if event['op'] == 'done':
            break
        call('submit', policy.handle(event))
    Path(args.output).write_text(json.dumps({'complete': True}) + '\n')


if __name__ == '__main__':
    main()
