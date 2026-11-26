#!/usr/bin/env python3
"""metrics_over_all3 MSE_ave1; strict interval coverage >= .93 in EVERY setting."""
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import key, read_predictions

def score(pred_path, labels_path):
    pred = read_predictions(pred_path)
    groups = {str(i): [] for i in range(1,6)}
    labels = {}
    with open(labels_path, newline='') as f:
        for row in csv.DictReader(f):
            k = key(row)
            if k in labels:
                raise ValueError('duplicate label key')
            setting, truth = row['setting'], float(row['truth'])
            if setting not in groups or not math.isfinite(truth):
                raise ValueError('invalid label')
            labels[k] = (setting, truth)
    if set(pred) != set(labels):
        raise ValueError('prediction/label key mismatch')
    for k, (setting, truth) in labels.items():
        mean, lower, upper = pred[k]
        # Original authors use strict inequalities, excluding interval endpoints.
        groups[setting].append(((mean-truth)**2, int(lower < truth < upper)))
    if any(not rows for rows in groups.values()):
        raise ValueError('all five settings required')
    mse = {s: sum(v[0] for v in rows)/len(rows) for s,rows in groups.items()}
    cov = {s: sum(v[1] for v in rows)/len(rows) for s,rows in groups.items()}
    raw = sum(mse.values())/5
    passed = all(v >= .93 for v in cov.values())
    return {'mean_mse': raw if passed else float('inf'), 'raw_mean_mse': raw,
            'coverage_by_setting': cov, 'mse_by_setting': mse, 'coverage_guard_passed': passed}

if __name__ == '__main__':
    try:
        result = score(sys.argv[1], sys.argv[2])
    except Exception as exc:
        result = {'mean_mse': float('inf'), 'coverage_guard_passed': False, 'invalid': str(exc)}
    print(json.dumps(result, sort_keys=True))
