"""Private predictable-bet execution, preserving the original capital experiment.

The candidate never receives the current observation before committing its bet.
Every permutation gets a fresh process and private writable filesystem.
"""
import json
import math
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import betting_interface as bi
from private_eval import Evaluator, remaining_time, validate_artifact

REQUEST_PROTOCOL = 'sta34-private-betting-v2'


def check_method(method):
    if not isinstance(method, dict) or type(method.get('version')) is not int or method['version'] != 2:
        raise ValueError('expected betting method version 2')
    execution = method.get('execution')
    if execution == 'certificate':
        if set(method) != {'version', 'execution', 'policy'}:
            raise ValueError('invalid certificate method keys')
        bi.check_policy(method['policy'])
    elif execution == 'stateful':
        if set(method) != {'version', 'execution', 'B', 'theta', 'trunc_scale', 'combine'}:
            raise ValueError('invalid stateful method keys')
        if type(method['B']) is not int or method['B'] < 1:
            raise ValueError('B must be a positive integer fixed before evaluation')
        for key in ('theta', 'trunc_scale'):
            if type(method[key]) not in (int, float) or not math.isfinite(method[key]):
                raise ValueError('invalid capital parameter')
        if not 0 <= method['theta'] <= 1 or not 0 < method['trunc_scale'] <= 1:
            raise ValueError('invalid capital bounds')
        if method['combine'] not in ('max', 'mixture'):
            raise ValueError('invalid capital combination')
    else:
        raise ValueError('unknown betting execution mode')
    return method


def validate_package(path):
    manifest = validate_artifact(path, 'Sta-34')
    seed = manifest['algorithm_seed']
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('invalid algorithm seed')
    if 'BENCH_SEED' in os.environ and seed != int(os.environ['BENCH_SEED']):
        raise ValueError('artifact seed differs from BENCH_SEED')
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo('model/betting.json')
        if info.file_size > 1024 * 1024:
            raise ValueError('method configuration is too large')
        method = check_method(json.loads(archive.read(info)))
    return manifest, method


class BettingReplay:
    """Commit two predictable bets, then reveal exactly one observation."""
    def __init__(self, x, method):
        self.x = np.asarray(x, dtype=np.float64)
        if self.x.ndim != 1 or not len(self.x) or not np.isfinite(self.x).all() or np.any((self.x < 0) | (self.x > 1)):
            raise ValueError('invalid bounded observations')
        self.method = method
        self.index = 0
        self.failed = False
        self.positive = 1.0
        self.negative = 1.0

    def __call__(self, operation, args, kwargs):
        try:
            if self.failed or operation != 'bet' or kwargs or len(args) != 1 or self.index >= len(self.x):
                raise ValueError('commit a bet before requesting an observation')
            bet = args[0]
            if not isinstance(bet, dict) or set(bet) != {'positive', 'negative'}:
                raise ValueError('expected positive and negative bets')
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in bet.values()):
                raise ValueError('bets must be finite real numbers')
            limit = 2 * self.method['trunc_scale']
            positive = min(limit, max(-limit, bet['positive']))
            negative = min(limit, max(-limit, bet['negative']))
            x = float(self.x[self.index])
            if self.method['theta'] > 0:
                self.positive *= 1 + positive * (x - 0.5)
            if self.method['theta'] < 1:
                self.negative *= 1 - negative * (x - 0.5)
            self.index += 1
            return {'x': x, 'done': self.index == len(self.x)}
        except Exception:
            self.failed = True
            raise

    def result(self):
        if self.failed or self.index != len(self.x):
            raise ValueError('incomplete or invalid predictable-bet transcript')
        # Match confseq's convention for cumulative capital that encounters 0*inf.
        positive = 0.0 if math.isnan(self.positive) else self.positive
        negative = 0.0 if math.isnan(self.negative) else self.negative
        theta = self.method['theta']
        if theta == 1:
            result = positive
        elif theta == 0:
            result = negative
        elif self.method['combine'] == 'mixture':
            result = theta * positive + (1 - theta) * negative
        else:
            result = max(theta * positive, (1 - theta) * negative)
        if not math.isfinite(result) or result < 0:
            raise ValueError('nonfinite or negative betting capital')
        return result


def stateful_instance(evaluator, x, iid, seed, method, request, affinity, legacy_seed=None):
    x = np.array(x, dtype=np.float64, copy=True)
    sizes = np.linspace(100, 2000, 10, dtype=int)
    matches = np.flatnonzero(sizes == len(x))
    if len(matches) != 1:
        raise ValueError('sample length is outside the fixed experiment')
    size_index = int(matches[0])
    base_seed = bi.algorithm_seed(iid, seed) if legacy_seed is None else int(legacy_seed)
    values = []
    for permutation in range(method['B']):
        remaining_time()
        # Exactly the original sequence of seeded in-place NumPy shuffles.
        np.random.RandomState((base_seed + size_index * permutation) % (2**32)).shuffle(x)
        replay = BettingReplay(x, method)
        outcome = evaluator.run(request, handler=replay, affinity=affinity)
        try:
            if json.loads(Path(outcome['output_path']).read_text()) != {'complete': True}:
                raise ValueError('invalid worker completion marker')
            values.append(replay.result())
        finally:
            # Do not retain a filesystem for every permutation in a full experiment.
            shutil.rmtree(Path(outcome['output_path']).parent)
    e_value = float(np.cumsum(values)[-1] / method['B'])
    if not math.isfinite(e_value):
        raise ValueError('nonfinite averaged betting capital')
    u = float(np.random.RandomState(base_seed).uniform(0, 1, 1)[-1])
    return e_value, u, int(e_value > u / 0.05)


def _certificate_job(job):
    x, iid, seed, policy, legacy_seed = job
    return bi.evaluate(x, iid, seed, policy, legacy_seed=legacy_seed)


def evaluate_artifact(pred_path, labels):
    remaining_time()
    available = sorted(os.sched_getaffinity(0))
    cpus = min(16, max(1, int(os.environ.get('BENCH_CPUS', len(available)))), len(available))
    with Evaluator(pred_path, 'Sta-34', cpus=cpus, timeout_s=None) as evaluator:
        # Validate the immutable snapshot that the evaluator extracted, not a racy caller path.
        method = check_method(json.loads((evaluator.package / 'model/betting.json').read_text()))
        seed = evaluator.manifest['algorithm_seed']
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('invalid algorithm seed')
        if 'BENCH_SEED' in os.environ and seed != int(os.environ['BENCH_SEED']):
            raise ValueError('artifact seed differs from BENCH_SEED')
        ids = [str(i) for i in labels['id']]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('invalid label IDs')
        legacy = labels.get('legacy_seed')
        if method['execution'] == 'certificate':
            jobs = ((labels['x'][k, :int(labels['n'][k])], iid, seed, method['policy'],
                     None if legacy is None else int(legacy[k])) for k, iid in enumerate(ids))
            with Pool(cpus) as pool:
                values = list(pool.imap(_certificate_job, jobs, chunksize=32))
        else:
            requests = {}
            for n in set(map(int, labels['n'])):
                request = evaluator.root / ('request-' + str(n) + '.json')
                request.write_text(json.dumps({'protocol': REQUEST_PROTOCOL, 'n': n,
                                               'null_mean': 0.5, 'alpha': 0.05}))
                requests[n] = request
            values = [None] * len(ids)
            def run_chunk(worker):
                for k in range(worker, len(ids), cpus):
                    n = int(labels['n'][k])
                    values[k] = stateful_instance(evaluator, labels['x'][k, :n], ids[k], seed,
                                                  method, requests[n], [available[worker]],
                                                  None if legacy is None else int(legacy[k]))
            with ThreadPoolExecutor(max_workers=cpus) as pool:
                list(pool.map(run_chunk, range(cpus)))
        remaining_time()
        return values, seed, method['execution']
