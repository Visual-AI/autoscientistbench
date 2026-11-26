#!/usr/bin/env python3
"""All-row yield R2 conditional on positive-class F1 >= 0.912 on test, >= 0.80 on val."""
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import read_prediction


F1_THRESHOLD = {'test': 0.912, 'val': 0.80}


def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'


def score(pred_path, labels_path):
    pred = read_prediction(pred_path)
    with open(labels_path, newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(pred):
        raise ValueError('Prediction and label IDs differ')
    y = [float(r['y']) for r in rows]
    if len(y) < 2 or not all(math.isfinite(v) for v in y):
        raise ValueError('Invalid labels')
    truth = [int(r['cls_label']) for r in rows]
    if any(c != int(v > 0) for c, v in zip(truth, y)):
        raise ValueError('Label inconsistency')
    tp = sum(c == 1 and pred[i][0] == 1 for c, i in zip(truth, ids))
    fp = sum(c == 0 and pred[i][0] == 1 for c, i in zip(truth, ids))
    fn = sum(c == 1 and pred[i][0] == 0 for c, i in zip(truth, ids))
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    avg = math.fsum(y) / len(y)
    den = math.fsum((v - avg)**2 for v in y)
    sse = math.fsum((v - pred[i][1])**2 for v, i in zip(y, ids))
    r2 = 1.0 - sse / den if den else (1.0 if sse == 0 else 0.0)
    # Infinity is the harness's invalid-submission sentinel for either direction.
    passed = f1 >= F1_THRESHOLD[split_of(labels_path)]
    return {'r2': r2 if passed else float('inf'), 'raw_r2': r2,
            'f1': f1, 'guard_passed': passed, 'tp': tp, 'fp': fp, 'fn': fn,
            'mae': math.fsum(abs(v - pred[i][1]) for v, i in zip(y, ids))/len(y),
            'rmse': math.sqrt(sse/len(y))}


if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as exc:
        result = {'r2': float('inf'), 'guard_passed': False, 'error': str(exc)}
    print(json.dumps(result))
