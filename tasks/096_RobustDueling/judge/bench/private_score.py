"""Replay causal policies and aggregate the original metric."""
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

TASK_ID = 'icml-741-RobustLinearDueling'
HEADER = ['id', 'actions', 'rounds', 'regret']
MAX_CPUS = 8


def cpu_groups(package, available):
    """Preserve optional intra-episode parallelism and the original total quota."""
    config = Path(package) / 'code/policy_resources.json'
    requested = 1
    if config.is_file():
        with config.open('rb') as stream:
            payload = stream.read(4097)
        if len(payload) > 4096:
            raise ValueError('policy resource configuration is too large')
        settings = json.loads(payload)
        if not isinstance(settings, dict) or set(settings) != {'episode_cpus'}:
            raise ValueError('policy resources must contain only episode_cpus')
        requested = settings['episode_cpus']
        if type(requested) is not int or not 1 <= requested <= MAX_CPUS:
            raise ValueError('episode_cpus must be an integer from 1 to 8')
    if not available:
        raise ValueError('no CPUs available for policy evaluation')
    available = available[:MAX_CPUS]
    if requested == 1:
        return [[cpu_id] for cpu_id in available]
    # The archived contract required single-threaded numerical libraries when
    # running multiple episode workers. A multicore episode therefore runs
    # alone, using at most the actual allocation; its algorithm stays editable.
    return [available[:min(requested, len(available))]]


def checked_record(output_path, row, handler):
    trusted = handler.accepted_record()
    candidate = read_pred(output_path)
    if set(candidate) != {row['id']}:
        raise ValueError('worker id mismatch')
    actions, rounds, regret = candidate[row['id']]
    if (actions != trusted['actions'] or rounds != trusted['rounds']
            or regret != trusted['regret']):
        raise ValueError('worker record differs from trusted finish')
    return dict(id=row['source_id'], **trusted)


def evaluate(pred, labels):
    # Input parsing belonged to the original predict command. It still counts;
    # only the new artifact preparation stage is excluded from that clock.
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
    cpu_ids = sorted(os.sched_getaffinity(0))[:MAX_CPUS]
    with tempfile.TemporaryDirectory(prefix='causal-inputs-') as tmp:
        tmp = Path(tmp)
        input_elapsed_s = time.monotonic() - input_started
        artifact_started = time.monotonic()
        with Evaluator(pred, TASK_ID, cpus=len(cpu_ids), timeout_s=None) as ev:
            groups = cpu_groups(ev.package, cpu_ids)[:len(rows)]
            artifact_prepare_s = time.monotonic() - artifact_started
            prediction_started = time.monotonic() - input_elapsed_s
            ev.started = prediction_started
            cpu_slots = queue.SimpleQueue()
            for group in groups:
                cpu_slots.put(group)

            def run(index_row):
                index, row = index_row
                group = cpu_slots.get()
                try:
                    inp = tmp / (str(index) + '.json')
                    inp.write_text(json.dumps({'id': row['id']}))
                    handler = Handler(row)
                    # All episodes use the same deadline; there was no separate
                    # per-episode time allowance in the original experiment.
                    result = ev.run(inp, handler=handler, affinity=group)
                    return checked_record(result['output_path'], row, handler)
                finally:
                    cpu_slots.put(group)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
                records = list(pool.map(run, enumerate(rows)))
            combined = tmp / 'trusted.csv'
            with combined.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=HEADER)
                writer.writeheader()
                for record in records:
                    row = dict(record)
                    row['actions'] = json.dumps(row['actions'], separators=(',', ':'))
                    writer.writerow(row)
            # The original predict command also serialized its output CSV.
            prediction_elapsed_s = time.monotonic() - prediction_started
            remaining_time()
        # Original metric reconstruction follows candidate execution.
        metrics = score(combined, labels / 'metric_labels.npz')
    if 'per_episode' in metrics:
        metrics['per_episode'] = {
            alias: metrics['per_episode'][source]
            for source, alias in zip(source_ids, config['ids'])}
    metrics.update(trusted_oracle=True, protocol_version=2,
                   verification_scope=config.get('scope', 'full'),
                   prediction_elapsed_s=prediction_elapsed_s,
                   artifact_prepare_s=artifact_prepare_s,
                   input_elapsed_s=input_elapsed_s,
                   worker_cpus=len(groups[0]), parallel_workers=len(groups))
    return metrics
