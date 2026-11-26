"""Trusted, fresh-process evaluation of each original rolling forecast window.

This module receives the private evaluation directory only in scorer. No whole
test-window array is passed to the candidate's outer predict or worker.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import stat
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import legacy_metric
from private_eval import Evaluator, validate_artifact, remaining_time


# Workers are isolated from each other, so windows can run side by side; start-up, not the forward pass, is most of a window's time.
MAX_WORKERS = 32


def configuration():
    return json.loads((HERE / 'forecast_config.json').read_text())


def one_window(inp, index, config):
    n = len(inp['id'])
    one = {}
    for key, values in inp.items():
        # Only the specified observation fields may reach the worker.
        if key not in {'id', 'horizon', 'x', 'channels', 'start', 'history_len'}:
            raise ValueError('unexpected private input field')
        one[key] = values[index:index + 1].copy() if key != 'channels' and values.ndim and len(values) == n else values.copy()
    # Real row ids never act as an implicit future-position lookup interface.
    one['id'] = np.array(['window'])
    if config['family'] == 'renf':
        length = min(int(one['history_len'][0]), int(config['max_lookback']))
        if length < 1:
            raise ValueError('empty allowed history')
        one['x'] = one['x'][:, -length:, :].copy()
        one['history_len'] = np.array([length], dtype=np.int64)
    elif one['x'].shape != (1, 96, 7):
        raise ValueError('expected exactly one 96-step forecast context')
    return one


def checked_npz(path):
    """Bound untrusted single-window results before NumPy allocates arrays."""
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('worker output must be a regular file')
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('worker output is too large for one window')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 16 or len({m.filename for m in members}) != len(members):
            raise ValueError('invalid prediction archive members')
        if sum(m.file_size for m in members) > 4 * 1024 * 1024:
            raise ValueError('expanded worker output is too large')
        for member in members:
            if '/' in member.filename or not member.filename.endswith('.npy'):
                raise ValueError('invalid prediction member path')
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('unsupported prediction NPY version')
                count = 1
                for dim in shape:
                    if not isinstance(dim, int) or dim < 0:
                        raise ValueError('invalid array dimension')
                    count *= dim
                if dtype.hasobject or count * dtype.itemsize > member.file_size or count * dtype.itemsize > 4 * 1024 * 1024:
                    raise ValueError('invalid or oversized prediction array')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def check_one(result, one, config):
    if config['family'] == 'renf':
        if set(result) != {'id', 'pred', 'seq_len', 'features'}:
            raise ValueError('invalid ReNF prediction keys')
        ids, forecast = result['id'], result['pred']
        seq_len, features = result['seq_len'], result['features']
        if seq_len.shape != () or seq_len.dtype.kind not in 'iu' or not 1 <= int(seq_len) <= min(720, int(one['history_len'][0])):
            raise ValueError('invalid lookback declaration')
        if features.shape != () or features.dtype.kind not in 'US' or str(features) != 'M':
            raise ValueError('features must equal M')
        horizon = 96
    else:
        horizon = int(one['horizon'][0])
        if set(result) != {f'id_{horizon}', f'pred_{horizon}'}:
            raise ValueError('unexpected horizon or prediction keys')
        ids, forecast = result[f'id_{horizon}'], result[f'pred_{horizon}']
    if ids.shape != (1,) or ids.dtype.kind not in 'US' or ids.tolist() != ['window']:
        raise ValueError('worker must predict only its supplied window')
    if forecast.shape != (1, horizon, 7) or forecast.dtype.kind != 'f' or not np.isfinite(forecast).all():
        raise ValueError('invalid one-window forecast')
    return horizon, forecast.copy()


def score(artifact, evaluation):
    config = configuration()
    validate_artifact(artifact, config['task_id'])
    evaluation = Path(evaluation)
    if not evaluation.is_dir():
        raise ValueError('protocol v2 requires an evaluation directory')
    prediction_started = time.monotonic()
    with np.load(evaluation / 'input.npz', allow_pickle=False) as archive:
        inp = {key: archive[key] for key in archive.files}
    if not len(inp['id']):
        raise ValueError('empty evaluation suite')
    cpus = min(int(config['cpus']), len(os.sched_getaffinity(0)))
    ids, predictions, seq_lengths = {}, {}, []
    with tempfile.TemporaryDirectory(prefix='forecast-trusted-') as folder:
        folder = Path(folder)
        artifact_started = time.monotonic()
        with Evaluator(artifact, config['task_id'], cpus=cpus,
                       timeout_s=None) as evaluator:
            artifact_setup_s = time.monotonic() - artifact_started
            workers = min(cpus, MAX_WORKERS)
            threads = cpus // workers
            slots = queue.SimpleQueue()
            for slot in range(workers):
                slots.put(slot)

            def run_window(allowed):
                slot = slots.get()
                try:
                    # The node's CPU pool can change during a run, so read it for every worker and retry if a CPU left in between.
                    for attempt in range(3):
                        available = sorted(os.sched_getaffinity(0))
                        affinity = [available[(slot * threads + k) % len(available)] for k in range(threads)]
                        try:
                            run = evaluator.run(allowed, affinity=affinity)
                        except ValueError as exc:
                            if 'invalid worker CPU allocation' not in str(exc) or attempt == 2:
                                raise
                            continue
                        if run['returncode'] == 0 or attempt == 2:
                            return run
                finally:
                    slots.put(slot)

            def evaluate(index):
                remaining_time()
                one = one_window(inp, index, config)
                allowed = folder / f'window-{index}.npz'
                with allowed.open('wb') as f:
                    np.savez(f, **one)
                run = run_window(allowed)
                allowed.unlink()
                if run['returncode'] != 0:
                    raise RuntimeError('isolated forecast worker failed')
                result = checked_npz(run['output_path'])
                horizon, forecast = check_one(result, one, config)
                return horizon, forecast, int(result['seq_len']) if config['family'] == 'renf' else None

            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                # map keeps window order, which the joined prediction arrays rely on
                for original_id, (horizon, forecast, seq_len) in zip(inp['id'], pool.map(evaluate, range(len(inp['id'])))):
                    ids.setdefault(horizon, []).append(original_id)
                    predictions.setdefault(horizon, []).append(forecast)
                    if seq_len is not None:
                        seq_lengths.append(seq_len)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        joined = folder / 'predictions.npz'
        if config['family'] == 'renf':
            if len(set(seq_lengths)) != 1:
                raise ValueError('lookback declaration must be consistent across windows')
            arrays = {'id': np.asarray(ids[96]), 'pred': np.concatenate(predictions[96]),
                      'seq_len': np.asarray(seq_lengths[0]), 'features': np.asarray('M')}
        else:
            arrays = {}
            for horizon in ids:
                arrays[f'id_{horizon}'] = np.asarray(ids[horizon])
                arrays[f'pred_{horizon}'] = np.concatenate(predictions[horizon])
        with joined.open('wb') as f:
            np.savez(f, **arrays)
        prediction_wall_s = time.monotonic() - prediction_started - artifact_setup_s
        remaining_time()
        if hasattr(legacy_metric, 'score'):
            result = legacy_metric.score(joined, evaluation / 'labels.npz')
        else:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                legacy_metric.main(joined, evaluation / 'labels.npz')
            result = json.loads(output.getvalue().strip().splitlines()[-1])
    result.update(protocol='private-forecast-v2', evaluation_windows=len(inp['id']),
                  isolation='fresh_worker_per_window', original_metric_preserved=True,
                  prediction_wall_s=prediction_wall_s, worker_pool_size=workers,
                  artifact_setup_s=artifact_setup_s)
    return result


def main():
    try:
        result = score(sys.argv[1], sys.argv[2])
    except Exception as exc:
        result = {'mse': float('inf'), 'protocol': 'private-forecast-v2',
                  'isolation_ok': False, 'reason': str(exc)}
    print(json.dumps(result))


if __name__ == '__main__':
    main()
