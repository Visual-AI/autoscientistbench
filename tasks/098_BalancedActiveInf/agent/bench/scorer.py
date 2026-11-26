"""Original active-inference metric with independent draw execution and queries."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from private_eval import Evaluator, validate_artifact, remaining_time
from output_checks import check
from legacy_metric import score as numeric_score

TASK_ID = 'imp-104-BalancedActiveInf'
WORKERS = 8
# R and rpy2 are not thread-safe: the schedule and every broker share this lock.
R_LOCK = threading.Lock()
CUBE = None


class CubeStream:
    """An R stream whose state is not shared with any other stream object.

    R itself has one process-global RNG. Restore/snapshot around every call so
    a label-dependent draw-local call cannot mutate the canonical next draw.
    Calls are serialized by R_LOCK, so concurrent draws cannot see each other's state.
    """
    def __init__(self, seed):
        with R_LOCK:
            from rpy2.robjects import r, numpy2ri, default_converter, IntVector
            from rpy2.robjects.conversion import localconverter
            from rpy2.robjects.packages import importr
            global CUBE
            # importr needs rpy2's conversion context, which only the main thread has: CubeSchedule makes the first stream there.
            if CUBE is None:
                CUBE = importr('BalancedSampling').cube
            self.cube = CUBE
        self.r = r
        self.conversion = numpy2ri
        self.default_converter = default_converter
        self.localconverter = localconverter
        self.IntVector = IntVector
        with R_LOCK, localconverter(default_converter + numpy2ri.converter):
            r['set.seed'](int(seed))
            self.state = np.asarray(r['.Random.seed'], dtype=np.int32).copy()

    def sample(self, auxiliary, probabilities):
        # Broker calls execute on its serving thread; conversion is context-local.
        with R_LOCK, self.localconverter(self.default_converter + self.conversion.converter):
            self.r['assign']('.Random.seed', self.IntVector(self.state))
            out = self.cube(self.conversion.py2rpy(auxiliary),
                            prob=self.conversion.py2rpy(probabilities))
            self.state = np.asarray(self.r['.Random.seed'], dtype=np.int32).copy()
            return (np.asarray(out).astype(int) - 1).tolist()


def draw_seed(algorithm_seed, draw):
    # Public deterministic scheduling, independent of labels and candidate calls.
    raw = hashlib.sha256(('active-draw-v1:%d:%d' % (algorithm_seed, draw)).encode()).digest()
    return int.from_bytes(raw[:8], 'big') % 2147483646 + 1


class CubeSchedule:
    """Advance exactly the original default sample, using only public arrays."""
    def __init__(self, algorithm_seed, predictions, meta):
        self.seed = int(algorithm_seed)
        self.stream = CubeStream(self.seed)
        u = predictions['uncertainty']
        self.pi = np.minimum(meta['tau'] * (u / u.mean()) * meta['budget']
                             + (1.0 - meta['tau']) * meta['budget'], 1.0)
        self.auxiliary = predictions['error_pred'].reshape(-1, 1)
        self.next_draw = 0

    def next(self, draw):
        if draw != self.next_draw:
            raise ValueError('draw schedule cannot be reset or reordered')
        # Always advance before running the candidate. Neither its design nor
        # the number/timing of its queries can change this stream's next state.
        selected = self.stream.sample(self.auxiliary, self.pi)
        self.next_draw += 1
        return (self.auxiliary, self.pi, selected), CubeStream(draw_seed(self.seed, draw))


class Oracle:
    def __init__(self, labels, stream, canonical=None):
        self.labels = labels
        self.stream = stream
        self.canonical = canonical
        self.canonical_used = False
        self.seen = set()
        self.cube_calls = 0

    def __call__(self, method, args, kwargs):
        if kwargs:
            raise ValueError('no keyword operations')
        if method == 'query':
            if len(args) != 1 or not isinstance(args[0], list):
                raise ValueError('indices required')
            ids = args[0]
            if (len(ids) > len(self.labels)
                    or any(type(i) is not int or i < 0 or i >= len(self.labels) for i in ids)
                    or len(set(ids)) != len(ids) or self.seen.intersection(ids)):
                raise ValueError('duplicate/out-of-range label request')
            self.seen.update(ids)
            return self.labels[ids].copy()
        if method == 'cube_sample':
            if len(args) != 2:
                raise ValueError('auxiliary and probabilities required')
            aux, pi = map(np.asarray, args)
            if aux.ndim == 1:
                aux = aux.reshape(-1, 1)
            # Intermediate sampling may use a subset and conditional 0/1
            # probabilities. Only the final reported population design is
            # subject to the original positive-pi / sum-pi guard below.
            if (aux.ndim != 2 or pi.ndim != 1 or len(pi) < 1
                    or aux.shape[0] != len(pi) or aux.shape[1] < 1
                    or aux.dtype.kind not in 'fiu' or pi.dtype.kind not in 'fiu'
                    or not np.isfinite(aux).all() or not np.isfinite(pi).all()
                    or np.any(pi < 0) or np.any(pi > 1)):
                raise ValueError('invalid cube dimensions/probabilities')
            self.cube_calls += 1
            if self.canonical is not None and not self.canonical_used:
                expected_aux, expected_pi, selected = self.canonical
                if (aux.dtype == expected_aux.dtype and pi.dtype == expected_pi.dtype
                        and np.array_equal(aux, expected_aux)
                        and np.array_equal(pi, expected_pi)):
                    self.canonical_used = True
                    return list(selected)
            # Multiple calls, including after label feedback, are legal within
            # this draw. This stream is discarded at its end.
            return self.stream.sample(aux, pi)
        raise ValueError('unsupported private operation')


def score(pred, root):
    root = Path(root)
    manifest = validate_artifact(pred, TASK_ID)
    meta = json.loads((root / 'labels.json').read_text())
    if meta['scope'] == 'full' and meta['n_replications'] != 10000:
        raise ValueError('original replication count required')
    for name, key in [('oracle.npy', 'oracle_sha256'), ('predictions.npz', 'predictions_sha256')]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != meta[key]:
            raise ValueError('private episode identity mismatch')
    truth = np.load(root / 'oracle.npy', allow_pickle=False)
    if truth.shape != (meta['population_size'],) or not np.isfinite(truth).all():
        raise ValueError('invalid private labels')
    with np.load(root / 'predictions.npz', allow_pickle=False) as z:
        predictions = {k: z[k].copy() for k in ('y_pred', 'uncertainty', 'error_pred')}
    designs, design_keys, records = {}, {}, []
    total_queries = total_cube_calls = total_cube_requests = 0
    n_draws = meta['n_replications']
    with tempfile.TemporaryDirectory(prefix='active-score-') as tmp:
        tmp = Path(tmp)
        unpack_started = time.monotonic()
        with Evaluator(pred, TASK_ID, cpus=8, timeout_s=None) as ev:
            unpack_s = time.monotonic() - unpack_started
            # Report prediction elapsed time; execution uses the overall deadline.
            started = time.monotonic()
            schedule = CubeSchedule(manifest['algorithm_seed'], predictions, meta)
            claim_lock = threading.Lock()
            cpu_lock = threading.Lock()
            held = set()
            state = {'next': 0, 'failed': False}
            results = [None] * n_draws

            def acquire_cpu():
                # The node's CPU pool can change during a run, so pick from the live mask.
                with cpu_lock:
                    live = os.sched_getaffinity(0)
                    cpu = min(live - held or live)
                    held.add(cpu)
                    return cpu

            def run_draw(inp, draw, public, canonical, stream):
                for attempt in range(5):
                    # A retry needs its own oracle and stream: the failed worker may have used them.
                    oracle = Oracle(truth, stream if attempt == 0 else CubeStream(public['draw_seed']), canonical)
                    cpu = acquire_cpu()
                    try:
                        run = ev.run(inp, handler=oracle, timeout_s=remaining_time(), affinity=[cpu],
                                     env={'BENCH_SMOKE': '1' if meta['scope'] == 'smoke' else '0',
                                          'BENCH_DRAW_INDEX': str(draw), 'BENCH_DRAW_SEED': str(public['draw_seed'])})
                    except ValueError as exc:
                        if 'invalid worker CPU allocation' not in str(exc) or attempt == 4:
                            raise
                        continue
                    finally:
                        with cpu_lock:
                            held.discard(cpu)
                    if run['returncode'] == 0 or cpu in os.sched_getaffinity(0) or attempt == 4:
                        return run, oracle

            def worker(slot):
                inp = tmp / f'input-{slot}'
                inp.mkdir()
                shutil.copy(root / 'predictions.npz', inp / 'predictions.npz')
                while True:
                    remaining_time()
                    # The original default stream has one order, so draws are claimed and advanced in it.
                    with claim_lock:
                        draw = state['next']
                        if draw >= n_draws or state['failed']:
                            return
                        state['next'] += 1
                        canonical, stream = schedule.next(draw)
                    try:
                        public = {k: meta[k] for k in ('version', 'population_size',
                                  'predictions_sha256', 'budget', 'tau', 'confidence')}
                        public.update(protocol='private-active-label-v2', task_id=TASK_ID,
                                      scope='private_draw', source_scope=meta['scope'], n_replications=1,
                                      total_replications=n_draws, draw_index=draw,
                                      algorithm_seed=manifest['algorithm_seed'],
                                      draw_seed=draw_seed(manifest['algorithm_seed'], draw))
                        (inp / 'manifest.json').write_text(json.dumps(public))
                        run, oracle = run_draw(inp, draw, public, canonical, stream)
                        out = Path(run['output_path'])
                        if out.stat().st_size > meta['population_size'] * 1000:
                            raise ValueError('oversized draw output')
                        piece, _ = check(out, public)
                        row = piece['trials'][0]
                        if set(row['selected']) != oracle.seen:
                            raise ValueError('reported selected units differ from actual label disclosures')
                        results[draw] = (piece, row, len(oracle.seen), oracle.cube_calls, oracle.canonical_used)
                    except BaseException:
                        state['failed'] = True
                        raise

            workers = min(WORKERS, len(os.sched_getaffinity(0)), n_draws)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in [pool.submit(worker, slot) for slot in range(workers)]:
                    future.result()
            # Design names and record order follow draw order, whatever order the workers finished in.
            for draw, (piece, row, seen, cube_calls, canonical_used) in enumerate(results):
                # Multiple-stage methods report their final design, not the pi
                # of an arbitrary intermediate cube call. Statistical validity
                # of that design/estimator remains subject to method review.
                used = piece['designs'][row['design']]
                key = json.dumps(used, sort_keys=True, separators=(',', ':'))
                if key not in design_keys:
                    name = 'design_' + str(len(designs))
                    design_keys[key] = name
                    designs[name] = used
                records.append({**row, 'id': draw, 'design': design_keys[key]})
                total_queries += seen
                total_cube_requests += cube_calls
                total_cube_calls += 1 + cube_calls - int(canonical_used)
            combined = tmp / 'combined.json'
            combined.write_text(json.dumps(dict(version=1, predictions_sha256=meta['predictions_sha256'],
                                                 designs=designs, trials=records), allow_nan=False))
            metric = numeric_score(combined, root / 'labels.json')
            remaining_time()
            wall_s = time.monotonic() - started
    return {**metric, 'protocol': 'private-active-label-v2', 'isolation_ok': True,
            'isolation': 'fresh_worker_per_draw', 'label_access_enforcement': 'private_rpc',
            'actual_queries': total_queries, 'actual_cube_calls': total_cube_calls,
            'cube_requests': total_cube_requests, 'canonical_cube_calls': meta['n_replications'],
            'predict_wall_s': wall_s,
            'artifact_unpack_s': unpack_s, 'worker_pool_size': workers,
            'randomness': 'original_default_R_stream; independent_draw_local_custom_streams',
            'method_audit_required': True}


if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as exc:
        result = {'mean_ci_width': float('inf'), 'eligible': False,
                  'isolation_ok': False, 'reason': str(exc)}
    print(json.dumps(result, allow_nan=True))
