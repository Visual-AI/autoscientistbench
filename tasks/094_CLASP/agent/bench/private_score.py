"""Replay policies causally and aggregate the original metric."""
import concurrent.futures
import csv
import json
import os
import queue
import tempfile
import time
from pathlib import Path

from private_eval import Evaluator,remaining_time
from private_handler import Handler
from legacy_validate import read_pred
from legacy_scorer import score

TASK_ID = 'icml-6078-CLASP'
HEADER = ['id', 'actions', 'rounds', 'loss', 'ccv1', 'ccv2']
MAX_CPUS = 8


def cpu_groups(package, available):
    """Allow an episode to use the original total CPU allocation."""
    config = Path(package) / 'code/policy_resources.json'
    requested = 1
    if config.is_file():
        with config.open('rb') as f:
            payload = f.read(4097)
        if len(payload) > 4096:
            raise ValueError('policy resource configuration is too large')
        settings = json.loads(payload)
        if not isinstance(settings, dict) or set(settings) != {'episode_cpus'}:
            raise ValueError('policy resources must contain only episode_cpus')
        requested = settings['episode_cpus']
        if type(requested) is not int or not 1 <= requested <= MAX_CPUS:
            raise ValueError('episode_cpus must be an integer from 1 to 8')
    if not available:
        raise RuntimeError('no CPU allocation for policy replay')
    count = min(requested, len(available))
    return [available[i:i + count]
            for i in range(0, len(available) - count + 1, count)]


def check_prediction_budget(started):
    elapsed = time.monotonic() - started
    remaining_time()
    return elapsed


def checked_record(output_path, episode_id, handler):
    trusted = handler.accepted_record()
    candidate = read_pred(output_path)
    if len(candidate) != 1 or candidate[0]['id'] != episode_id:
        raise ValueError('worker id mismatch')
    if (candidate[0]['actions'] != trusted['actions']
            or int(candidate[0]['rounds']) != trusted['rounds']
            or any(candidate[0][k] != trusted[k] for k in ('loss', 'ccv1', 'ccv2'))):
        raise ValueError('worker record differs from trusted finish')
    return trusted


def evaluate(pred, labels):
    input_started = time.monotonic()
    labels = Path(labels)
    config = json.loads((labels / 'manifest.json').read_text())
    if config['task_id'] != TASK_ID or config['protocol'] != 'private-policy-v2':
        raise ValueError('wrong private episode container')
    with (labels / 'input.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    source_ids = config.get('source_ids', config['ids'])
    if (not rows or [r['id'] for r in rows] != source_ids
            or len(set(config['ids'])) != len(rows)
            or len(config['ids']) != len(rows)):
        raise ValueError('private episode id mismatch')
    rows = [dict(r, source_id=r['id'], id=alias)
            for r, alias in zip(rows, config['ids'])]
    available = sorted(os.sched_getaffinity(0))[:MAX_CPUS]
    if not available:
        raise RuntimeError('no CPU allocation for policy replay')
    with tempfile.TemporaryDirectory(prefix='causal-inputs-') as tmp:
        tmp = Path(tmp)
        input_elapsed = time.monotonic() - input_started
        artifact_started = time.monotonic()
        with Evaluator(pred, TASK_ID, cpus=len(available), timeout_s=None) as ev:
            groups = cpu_groups(ev.package, available)
            workers = min(len(groups), len(rows))
            cpu_pool = queue.Queue()
            for group in groups[:workers]:
                cpu_pool.put(group)
            artifact_prepare_wall_s = time.monotonic() - artifact_started
            # Only the new artifact freeze/validation/extraction is infrastructure.
            # Original input preparation, episodes, and final output serialization
            # share one wall clock; workers never receive a new per-episode cap.
            ev.started = time.monotonic() - input_elapsed
            check_prediction_budget(ev.started)

            def run(index_row):
                index, row = index_row
                group = cpu_pool.get()
                try:
                    check_prediction_budget(ev.started)
                    inp = tmp / (str(index) + '.json')
                    inp.write_text(json.dumps({'id': row['id']}))
                    handler = Handler(row)
                    result = ev.run(inp, handler=handler, affinity=group)
                    trusted = checked_record(result['output_path'], row['id'], handler)
                    return dict(id=row['source_id'], **trusted)
                finally:
                    cpu_pool.put(group)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(run, enumerate(rows)))
            combined = tmp / 'trusted.csv'
            with combined.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                for record in records:
                    writer.writerow(dict(record, actions=json.dumps(
                        record['actions'], separators=(',', ':'))))
            prediction_wall_s = check_prediction_budget(ev.started)
        metrics = score(combined, labels / 'metric_labels.npz')
    metrics.update(trusted_oracle=True, protocol_version=2,
                   verification_scope=config.get('scope', 'full'),
                   prediction_wall_s=prediction_wall_s,
                   prediction_cpus=sum(len(group) for group in groups[:workers]),
                   worker_count=workers, cpus_per_worker=len(groups[0]),
                   artifact_prepare_wall_s=artifact_prepare_wall_s)
    return metrics
