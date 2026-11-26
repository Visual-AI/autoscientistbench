#!/usr/bin/env python3
"""Mean accuracy (%) of five independent replicas; never ensemble their decisions.
Artifact integrity is checked by check_model.py; independent training and data use require patch audit.
Infinity is the harness rejection sentinel (the paper-rule value on rejection is zero).
"""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import read_predictions

def score(pred, labels):
    ids, p = read_predictions(pred)
    with open(labels, newline='') as f:
        r = csv.reader(f)
        if next(r, None) != ['id', 'y']: raise ValueError('invalid labels header')
        rows = list(r)
    truth_ids = np.asarray([int(row[0]) for row in rows])
    y = np.asarray([int(row[1]) for row in rows])
    if not np.array_equal(ids, truth_ids) or len(np.unique(truth_ids)) != len(truth_ids):
        raise ValueError('prediction and truth id mismatch')
    if not np.all((y >= 0) & (y <= 9)): raise ValueError('invalid truth')
    correct = (p == y[:, None]).sum(0)
    per_replica = correct / len(y) * 100
    return {'accuracy': float(per_replica.mean()), 'per_replica_accuracy': per_replica.tolist(),
            'correct_per_replica': correct.tolist(), 'n': len(y)}
if __name__ == '__main__':
    try: result = score(sys.argv[1], sys.argv[2])
    except Exception as e: result = {'accuracy': float('inf'), 'rule_score': 0, 'guard_failed': str(e)}
    print(json.dumps(result))
