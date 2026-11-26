#!/usr/bin/env python3
"""Separate private fit/certification workers; trusted held-out FFT reconstruction."""
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import Evaluator, remaining_time
from certificate import TAU, training_grid, check_candidates, selected_tree, decisions, risks, aggregate

TASK_ID = 'icml-3290-LTARC'
PROTOCOL = 'private-policy-v2'
FORMAL = 'star-original-100'
DEVELOPMENT = 'ist-public-development-v1'
# suite: rows, covariate columns, repetitions, treatment-risk tolerance.
# Development is the paper's International Stroke Trial data, which shares no rows with STAR.
POPULATION = {FORMAL: (4218, 18, 100, TAU), DEVELOPMENT: (18298, 53, 40, .225)}


def load_cohort(path):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'x', 'a', 'y', 'names', 'boolean', 'roles', 'suite'}:
            raise ValueError('invalid cohort')
        data = {k: z[k] for k in z.files}
    n, features = data['x'].shape
    suite = str(data['suite'])
    if suite not in POPULATION:
        raise ValueError('unknown evaluation suite')
    rows, columns, repetitions, data['tau'] = POPULATION[suite]
    if (n, features) != (rows, columns):
        raise ValueError('original population required')
    if data['roles'].shape != (repetitions, n) or data['a'].shape != (n,) or data['y'].shape != (n,):
        raise ValueError('cohort dimensions differ')
    if data['names'].shape != (features,) or data['boolean'].shape != (features,):
        raise ValueError('invalid feature schema')
    if data['boolean'].dtype != bool or not np.isfinite(data['x']).all():
        raise ValueError('invalid feature values')
    if not np.isin(data['x'][:, data['boolean']], (0, 1)).all():
        raise ValueError('invalid Boolean features')
    if not np.isin(data['a'], (0, 1)).all() or not np.isin(data['y'], (0, 1)).all():
        raise ValueError('invalid factual labels')
    for i, roles in enumerate(data['roles']):
        expected = np.concatenate([np.zeros(int(n * .5), dtype=np.int8),
                                   np.ones(int(n * .25), dtype=np.int8),
                                   np.full(n - int(n * .5) - int(n * .25), 2, dtype=np.int8)])
        np.random.RandomState(10603 + i).shuffle(expected)
        if not np.array_equal(roles, expected):
            raise ValueError('original partition protocol required')
    return data


def write_phase(path, phase, data, mask, criteria=None, candidates=None):
    path.mkdir()
    request = {'task': TASK_ID, 'phase': phase, 'tau': data['tau'], 'n_bins': 40,
               'max_splits': 3, 'gamma': 1, 'propensity': .5,
               'names': data['names'].tolist(), 'boolean': data['boolean'].tolist()}
    if criteria is not None:
        request['criteria'] = criteria
        request['candidates'] = candidates
    (path / 'request.json').write_text(json.dumps(request))
    np.savez(path / 'data.npz', x=data['x'][mask], a=data['a'][mask], y=data['y'][mask])


def json_output(path):
    if Path(path).stat().st_size > 64 * 1024 * 1024:
        raise ValueError('certificate too large')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate certificate key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite certificate value')
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def evaluate_fold(ev, data, i, root, affinity):
    roles = data['roles'][i]
    grid = training_grid(data['x'][roles == 0], data['names'], data['boolean'])
    fit = root / ('fit-' + str(i))
    write_phase(fit, 'fit', data, roles == 0)
    result = ev.run(fit, affinity=affinity)
    # Parse and validate before any certification outcomes are released.
    candidates = check_candidates(json_output(result['output_path']), grid)
    certify = root / ('certify-' + str(i))
    write_phase(certify, 'certify', data, roles == 1, grid, candidates)
    selected = ev.run(certify, affinity=affinity)
    tree = selected_tree(json_output(selected['output_path']), candidates)
    mask = roles == 2
    d = decisions(tree, data['x'][mask], grid)
    remaining_time()
    return risks(d, data['a'][mask], data['y'][mask])


def score(pred_path, labels_path):
    """Return only aggregate metrics or a fixed redacted failure reason."""
    try:
        data = load_cohort(labels_path)
        affinity = [min(os.sched_getaffinity(0))]
        values = []
        with tempfile.TemporaryDirectory(prefix='ltarc-score-') as tmp:
            root = Path(tmp)
            resources = root / 'empty-resources'
            resources.mkdir()
            with Evaluator(pred_path, TASK_ID, cpus=1, data_dir=resources) as ev:
                for i in range(len(data['roles'])):
                    values.append(evaluate_fold(ev, data, i, root, affinity))
        result = aggregate(values, data['tau'])
        result.update(protocol=PROTOCOL, suite=str(data['suite']), repetitions=len(values),
                      certificate_verified=True, private_roles_verified=True,
                      development=str(data['suite']) == DEVELOPMENT)
        return result
    except Exception:
        # Evaluator exceptions can contain arbitrary candidate stderr/private rows.
        # Never return str(exc), traceback, worker stdout or per-fold trees here.
        return {'population_risk': float('inf'), 'guard_pass': False,
                'certificate_verified': False, 'private_roles_verified': False,
                'reason': 'private replay or certificate validation failed'}


if __name__ == '__main__':
    print(json.dumps(score(*sys.argv[1:])))
