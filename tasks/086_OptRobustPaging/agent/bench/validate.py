#!/usr/bin/env python3
"""Check the sealed policy artifact and the public request without executing candidate code."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_eval import validate_artifact

TASK_ID = 'icml-4413-OptimalRobustnessLearning'

if __name__ == '__main__':
    try:
        request = json.loads(Path(sys.argv[2]).read_text())
        if request != {'task': TASK_ID, 'protocol': 'private-policy-v2'}:
            raise ValueError('wrong public request')
        validate_artifact(sys.argv[1], TASK_ID)
        print('ok: sealed policy artifact; misses are counted by the trusted replay')
    except Exception as exc:
        print('invalid: ' + str(exc), file=sys.stderr)
        sys.exit(1)
