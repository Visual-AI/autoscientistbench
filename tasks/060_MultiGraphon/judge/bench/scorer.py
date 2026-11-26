#!/usr/bin/env python3
"""MISE x1000: equal graphon weight, then equal trial weight.
Invalid predictions return Infinity. Timing authenticity is a separate judge probe / patch audit.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import read_npz, read_prediction
from metric import mise_step


def evaluate(pred_path, labels_path):
    lab = read_npz(labels_path)
    pred = read_prediction(pred_path, lab['id'].tolist())
    groups = lab['graphon_id']
    required = set(range(11)) | {12}
    if set(groups.tolist()) != required:
        raise ValueError('labels must cover the twelve graphons')
    refs = {int(g): w for g, w in zip(lab['reference_id'], lab['reference'])}
    errors, times = [], []
    for iid, g in zip(lab['id'], groups):
        w, seconds = pred[str(iid)]
        errors.append(1000 * mise_step(w, refs[int(g)]))
        times.append(seconds)
    errors = np.array(errors)
    per_group = [errors[groups == g] for g in sorted(required)]
    raw = float(np.mean([v.mean() for v in per_group]))
    mc_se = float(np.sqrt(sum(v.var(ddof=1) / len(v) if len(v)>1 else 0 for v in per_group)) / len(required))
    avg_time = float(np.mean(times))
    return {'mise': raw, 'mise_raw': raw,
            'mean_time_s': avg_time,
            'mc_se': mc_se, 'per_graphon': {str(g+1): float(errors[groups == g].mean()) for g in sorted(required)}}


if __name__ == '__main__':
    try:
        result = evaluate(sys.argv[1], sys.argv[2])
    except Exception as e:
        result = {'mise': float('inf'), 'invalid': str(e)}
    print(json.dumps(result))
