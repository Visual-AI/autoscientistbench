#!/usr/bin/env python3
"""Pooled sklearn Cohen kappa over every fixed QC cell, versus three-annotator majority.
All-different annotation ties were resolved to Annotator 1 when labels were materialized.
Missing/out-of-class labels are separate sentinels on the two sides: always mismatches.
Malformed/duplicate/extra IDs are protocol failures (Infinity).
"""
import json
import math
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_io import read_table


def score(pred_path, labels_path):
    truth = read_table(labels_path)
    if not truth:
        raise ValueError('Empty ground truth')
    pred = read_table(pred_path)
    if set(pred) - set(truth):
        raise ValueError('Extra cell IDs')
    ids = sorted(truth)
    y = np.array([truth[i] if truth[i] in range(7) else -2 for i in ids])
    p = np.array([pred.get(i, -1) if pred.get(i, -1) in range(7) else -1 for i in ids])
    # Numeric encoding is a bijective relabeling of the seven original class strings.
    value = float(cohen_kappa_score(p, y))
    if not math.isfinite(value):
        raise ValueError('Undefined kappa')
    return {'pooled_kappa': value, 'n_cells': len(ids), 'n_missing': len(set(truth)-set(pred)),
            'n_invalid_labels': int(np.sum(p == -1)), 'accuracy': float(np.mean(p == y))}


if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as exc:
        result = {'pooled_kappa': float('inf'), 'invalid': str(exc)}
    print(json.dumps(result))
