#!/usr/bin/env python3
"""Trusted chronological execution and original mean relative link-count error."""
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import Evaluator
from replay import Replay
TASK_ID = 'Fin-03'

def score(pred_path, labels_path):
    input_started = time.monotonic()
    private = Path(labels_path)
    replay = Replay(private / 'input.csv', private / 'labels.csv')
    input_elapsed = time.monotonic() - input_started
    with tempfile.TemporaryDirectory(prefix='fin03-score-') as directory:
        request = Path(directory) / 'request.json'
        request.write_text(json.dumps({'task': TASK_ID, 'protocol': 'private-policy-v2'}))
        with Evaluator(pred_path, TASK_ID, cpus=8, timeout_s=None) as evaluator:
            artifact_elapsed = time.monotonic() - evaluator.started
            # Report input/policy and inert-artifact preparation separately.
            prediction_started = time.monotonic() - input_elapsed
            evaluator.started = prediction_started
            result = evaluator.run(request, handler=replay)
            if result['returncode'] != 0:
                raise ValueError('isolated policy failed')
            prediction_elapsed = time.monotonic() - prediction_started
        metrics = replay.result()
        metrics['policy_elapsed_s'] = result['elapsed_s']
        metrics['prediction_elapsed_s'] = prediction_elapsed
        metrics['artifact_prepare_s'] = artifact_elapsed
        return metrics

if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as exc:
        result = {'re_l': float('inf'), 'format_guard_ok': False,
                  'causality_guard_ok': False, 'reason': str(exc)}
    print(json.dumps(result))
