"""Macro recall with the binding pooled/per-cell null constraints.

Pattern agnosticism requires patch audit; it cannot be inferred from calls.
The explicit task rule assigns ZERO to a failed statistical guard.
"""
import json
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate import read_predictions


def score(pred_path, label_path):
    pred = read_predictions(pred_path)
    with np.load(label_path, allow_pickle=False) as z:
        lab = {k: z[k] for k in z.files}
    ids = lab['id'].tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(pred):
        raise ValueError('prediction/label ID mismatch')
    calls = np.array([pred[i] for i in ids], dtype=float)
    scenarios, levels, reps = lab['scenario'], lab['n_cells'], lab['rep']
    groups = {}
    exact_power = []
    clustered = {}
    null_calls = []
    for s in np.unique(scenarios):
        ix = scenarios == s
        for n in np.unique(levels[ix]):
            pick = ix & (levels == n)
            groups[f'{s}/{n}'] = float(calls[pick].mean())
            if not s.startswith('null_'):
                exact_power.append(Fraction(int(calls[pick].sum()), int(pick.sum())))
        cr = [float(calls[ix & (reps == r)].mean()) for r in np.unique(reps[ix])]
        clustered[s] = float(np.var(cr, ddof=1)/len(cr)) if len(cr)>1 else 0.
        if s.startswith('null_'):
            null_calls.extend(calls[ix].tolist())
    power = [v for k, v in groups.items() if not k.startswith('null_')]
    null = [v for k, v in groups.items() if k.startswith('null_')]
    if not power or not null:
        raise ValueError('both pattern and null labels are required')
    raw = float(sum(exact_power) / len(exact_power))
    pooled, maximum = float(np.mean(null_calls)), max(null)
    guards_ok = pooled <= .058 and maximum <= .094
    pattern_vars = [v for s,v in clustered.items() if not s.startswith('null_')]
    se = float(np.sqrt(sum(pattern_vars))/len(pattern_vars))
    return {'macro_recall': raw if guards_ok else 0., 'macro_recall_raw': raw,
            'macro_recall_se': se, 'null_reject_pooled': pooled,
            'null_reject_max_cell': maximum, 'null_pooled_limit': .058,
            'null_cell_limit': .094, 'statistical_guards_ok': bool(guards_ok),
            'pattern_agnostic_enforcement': 'patch audit; subset/ID probe is partial evidence',
            'recall_by_cell': {k:v for k,v in groups.items() if not k.startswith('null_')},
            'null_reject_by_cell': {k:v for k,v in groups.items() if k.startswith('null_')}}


if __name__ == '__main__':
    try:
        result = score(*sys.argv[1:])
    except Exception as e:
        result = {'macro_recall': float('inf'), 'error': str(e)}
    print(json.dumps(result))
