#!/usr/bin/env python3
"""Chronological private replay followed by unchanged CAST accounting."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from accounting import evaluate, COLS
from private_eval import Evaluator
from replay import PANELS, Replay, read_panels
TASK_ID = 'Fin-04'

def clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value

def score(pred_path, labels_path):
    input_started = time.monotonic()
    private = Path(labels_path)
    panels = read_panels(private / 'input.csv')
    input_elapsed = time.monotonic() - input_started
    all_orders, elapsed = [], 0.
    with tempfile.TemporaryDirectory(prefix='fin04-score-') as directory:
        directory = Path(directory)
        with Evaluator(pred_path, TASK_ID, cpus=16, timeout_s=None) as evaluator:
            artifact_elapsed = time.monotonic() - evaluator.started
            # Report inert-artifact preparation separately from policy execution.
            prediction_started = time.monotonic() - input_elapsed
            evaluator.started = prediction_started
            def run_panel(panel):
                replay = Replay(panel, panels[panel])
                request = directory / (panel + '.json')
                request.write_text(json.dumps({'task': TASK_ID, 'protocol': 'private-policy-v2',
                                               'panel': panel}))
                result = evaluator.run(request, handler=replay)
                if result['returncode'] != 0:
                    raise ValueError('isolated policy failed')
                return replay.result(), result['elapsed_s']
            # All workers share the same 16-core affinity. Independent panels
            # run concurrently as in the original predictor, without sharing
            # candidate state or observations across panel workers.
            with ThreadPoolExecutor(max_workers=min(4, len(os.sched_getaffinity(0)))) as pool:
                for orders, seconds in pool.map(run_panel, PANELS):
                    all_orders.extend(orders)
                    elapsed += seconds
        trusted_orders = directory / 'orders.csv'
        with trusted_orders.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(COLS)
            writer.writerows([[identifier, *values] for identifier, values in all_orders])
        prediction_elapsed = time.monotonic() - prediction_started
        result = evaluate(trusted_orders, private / 'labels.npz')
        result.update(causality_guard_ok=True, protocol='private-policy-v2',
                      policy_elapsed_s=elapsed, trusted_decisions=len(all_orders),
                      prediction_elapsed_s=prediction_elapsed,
                      artifact_prepare_s=artifact_elapsed)
        return result

if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as exc:
        result = dict(sharpe=float('inf'), guards_ok=False, causality_guard_ok=False,
                      failures=[str(exc)])
    print(json.dumps(clean(result)))
