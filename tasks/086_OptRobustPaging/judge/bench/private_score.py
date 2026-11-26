"""Authoritative replay of a sealed policy over every trace, and the paper's cost-ratio metric."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
import threading
import time

import numpy as np

from belady import lines_of, opt_misses
from private_eval import Evaluator, remaining_time
from private_paging import TraceReplay

TASK_ID = 'icml-4413-OptimalRobustnessLearning'
PROTOCOL = 'private-policy-v2'
MAX_WORKERS = 4


def load_traces(labels):
    labels = Path(labels)
    manifest = json.loads((labels / 'manifest.json').read_text())
    if manifest.get('task') != TASK_ID or manifest.get('protocol') != PROTOCOL:
        raise ValueError('wrong private trace container')
    traces = []
    for trace_id in manifest['traces']:
        with np.load(labels / (trace_id + '.npz'), allow_pickle=False) as z:
            traces.append((trace_id, z['pc'].tolist(), z['addr'].tolist()))
    if not traces:
        raise ValueError('empty trace list')
    return traces


def _worker_count(n_traces):
    declared = int(os.environ.get('OMP_NUM_THREADS') or MAX_WORKERS)
    return max(1, min(MAX_WORKERS, declared, n_traces, len(os.sched_getaffinity(0))))


def evaluate(pred_path, labels_path):
    started = time.monotonic()
    traces = load_traces(labels_path)
    opt_started = time.monotonic()
    optima = [opt_misses(lines_of(addresses)) for _, _, addresses in traces]
    opt_elapsed = time.monotonic() - opt_started
    if any(m <= 0 for m in optima):
        raise ValueError('empty optimum')
    workers = _worker_count(len(traces))
    lock = threading.Lock()
    held = set()

    def acquire():
        # The node's CPU pool can change during a run, so each worker picks from the live mask.
        with lock:
            live = os.sched_getaffinity(0)
            cpu = min(live - held or live)
            held.add(cpu)
            return cpu

    with tempfile.TemporaryDirectory(prefix='paging-score-') as scratch:
        request = Path(scratch) / 'request.json'
        request.write_text(json.dumps({'task': TASK_ID, 'protocol': PROTOCOL}))
        with Evaluator(pred_path, TASK_ID, cpus=workers, timeout_s=None) as evaluator:
            artifact_elapsed = time.monotonic() - evaluator.started

            def run(index):
                trace_id, pcs, addresses = traces[index]
                for attempt in range(5):
                    # A replay is consumed by its worker, so a retry needs a fresh one.
                    replay = TraceReplay(pcs, addresses)
                    cpu = acquire()
                    try:
                        result = evaluator.run(request, handler=replay, affinity=[cpu])
                    except ValueError as exc:
                        if 'invalid worker CPU allocation' not in str(exc) or attempt == 4:
                            raise
                        continue
                    finally:
                        with lock:
                            held.discard(cpu)
                    record = replay.accepted_record()
                    with Path(result['output_path']).open() as f:
                        submitted = json.loads(f.read())
                    if submitted != {'accesses': record['accesses'], 'misses': record['misses']}:
                        raise ValueError('candidate output differs from the trusted count')
                    shutil.rmtree(Path(result['output_path']).parent)
                    return {'id': trace_id, 'accesses': record['accesses'], 'misses': record['misses'],
                            'opt_misses': optima[index], 'cost_ratio': record['misses'] / optima[index],
                            'hit_ratio': 1.0 - record['misses'] / record['accesses'],
                            'policy_elapsed_s': result['elapsed_s'], 'rpc_calls': result['rpc_calls']}

            with ThreadPoolExecutor(max_workers=workers) as pool:
                per_trace = list(pool.map(run, range(len(traces))))
            remaining_time()
    ratios = [row['cost_ratio'] for row in per_trace]
    return {'mean_cost_ratio': sum(ratios) / len(ratios), 'mean_hit_ratio': sum(r['hit_ratio'] for r in per_trace) / len(ratios),
            'per_trace': per_trace, 'trusted_replay': True, 'protocol': PROTOCOL, 'worker_pool_size': workers,
            'opt_elapsed_s': opt_elapsed, 'artifact_prepare_s': artifact_elapsed,
            'scorer_elapsed_s': time.monotonic() - started}
