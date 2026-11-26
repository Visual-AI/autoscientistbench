#!/usr/bin/env python3
"""Replay the sealed policy in the trusted paging environment; prints one JSON line."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_score import evaluate

if __name__ == '__main__':
    try:
        result = evaluate(sys.argv[1], sys.argv[2])
    except Exception as exc:
        result = {'mean_cost_ratio': float('inf'), 'trusted_replay': False, 'reason': type(exc).__name__ + ': ' + str(exc)}
    print(json.dumps(result))
