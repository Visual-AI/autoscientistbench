#!/usr/bin/env python3
"""Cumulative squared-error regret, recomputed from estimates and hidden means.
For multiple independent development episodes, report mean endpoint regret.
No success, alarm-count, or baseline-relative constraint is imposed.
"""
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from numeric_validate import read_predictions


def score(pred_path, labels_path):
    ids, pred = read_predictions(pred_path)
    with open(labels_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['id','episode','t','mu']:
            raise ValueError('bad labels header')
        labels = list(reader)
    if ids != [r['id'] for r in labels]:
        raise ValueError('label ids or order mismatch')
    groups = {}
    for p, r in zip(pred, labels):
        mu = float(r['mu'])
        if not math.isfinite(mu):
            raise ValueError('nonfinite label')
        group = groups.setdefault(r['episode'], [])
        if int(r['t']) != len(group) + 1:
            raise ValueError('bad time sequence')
        group.append((p - mu)**2)
    endpoints = {k: float(np.sum(v)) for k, v in groups.items()}
    value = float(np.mean(list(endpoints.values())))
    if not math.isfinite(value):
        raise ValueError('nonfinite regret')
    return {'cumulative_regret':value, 'per_episode':endpoints, 'valid':True}

if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as e:
        result = {'cumulative_regret':float('inf'), 'valid':False, 'reason':str(e)}
    print(json.dumps(result))
