"""Frozen C2ST metric and quality/budget guards; prints one JSON line."""
import os
# Avoid oversubscribed BLAS in the independent classifier folds.
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
import json
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2st import c2st

_pairs = {}


def fold(job):
    ident, index = job
    torch.set_num_threads(1)
    threadpool_limits(1)
    ref, sample = _pairs[ident]
    return ident, index, c2st(ref, sample, fold_index=index)


def aggregate(values, meta, calls):
    groups = {task: [] for task in ('task1', 'task2', 'task3', 'task4', 'task5')}
    violations = []
    for ident, value in values.items():
        groups[meta['instances'][ident]['task']].append(value)
        if calls[ident] > 10000 * 100000:
            violations.append('likelihood budget exceeded')
    assert all(groups.values()), 'every task family is required'
    means = {task: float(np.mean(v)) for task, v in groups.items()}
    guard_tasks = ('task1', 'task2', 'task4', 'task5')
    guard_mean = float(np.mean([means[t] for t in guard_tasks]))
    if guard_mean > .5234:
        violations.append('other-task mean exceeds 0.5234')
    for task in guard_tasks:
        if means[task] > meta['baseline'][task] + .03:
            violations.append(task + ' exceeds baseline + 0.03')
    return {'c2st': float('inf') if violations else means['task3'],
            'protocol_worst': 1.0 if violations else None,
            'per_task': means, 'other_task_mean': guard_mean,
            'per_instance': values, 'max_evaluations_per_sample': max(calls.values()) / 10000,
            'violations': violations}


def score(pred_path, labels_path):
    global _pairs
    torch.set_num_threads(1)
    threadpool_limits(1)
    with np.load(labels_path, allow_pickle=False) as labels, np.load(pred_path, allow_pickle=False) as pred:
        meta = json.loads(str(labels['__meta__'].item()))
        ids = set(meta['instances'])
        assert set(labels.files) == ids | {'__meta__'}
        assert len(pred.files) == len(set(pred.files))
        assert set(pred.files) == ids | {i + '__calls' for i in ids}, 'prediction ids'
        calls = {}
        _pairs = {}
        for ident in sorted(ids):
            arr, ref = pred[ident], labels[ident]
            assert arr.shape == ref.shape and arr.shape[0] == 10000
            assert arr.dtype.kind == 'f' and np.isfinite(arr).all()
            count = pred[ident + '__calls']
            assert count.shape == () and count.dtype.kind in 'iu' and int(count) >= 0
            calls[ident] = int(count)
            _pairs[ident] = (torch.from_numpy(ref.copy()), torch.from_numpy(arr.copy()))
    if max(calls.values()) > 1000000000:
        return {'c2st': float('inf'), 'protocol_worst': 1.0, 'violations': ['likelihood budget exceeded']}
    jobs = [(i, f) for i in sorted(ids) for f in range(5)]
    workers = min(int(os.environ.get('BENCH_CPUS', os.environ.get('MAX_JOBS', '64'))),
                  len(os.sched_getaffinity(0)), len(jobs))
    values = {i: [None] * 5 for i in ids}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        for ident, index, value in pool.map(fold, jobs):
            values[ident][index] = value
    values = {i: float(np.asarray(np.mean(v)).astype(np.float32)) for i, v in values.items()}
    return aggregate(values, meta, calls)


if __name__ == '__main__':
    try:
        result = score(sys.argv[1], sys.argv[2])
    except Exception as exc:
        result = {'c2st': float('inf'), 'protocol_worst': 1.0, 'violations': [str(exc)]}
    print(json.dumps(result, allow_nan=True))
