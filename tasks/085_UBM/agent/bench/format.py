"""Frozen tabular output schema and observed-data layout."""
import csv
import numpy as np

SIGNALS = ('Y0', 'Y1', 'A', 'S')
MODEL_KEYS = ('S', 'A', 'Y0', 'Y1')
PROB_COLS = [f'{cohort}_{key}_{g:02d}' for cohort in ('rct', 'obs') for key in MODEL_KEYS for g in range(32)]
STAT_COLS = [f'SE_{key}_{suffix}' for key in SIGNALS for suffix in ('r', 'p')]
COLS = ['id'] + PROB_COLS + STAT_COLS


def read_prediction(path, expected_ids=None):
    rows = {}
    with open(path, newline='') as f:
        reader = csv.reader(f)
        if next(reader, None) != COLS:
            raise ValueError('wrong CSV header')
        for row in reader:
            if len(row) != len(COLS):
                raise ValueError('wrong number of fields')
            iid = row[0]
            if not iid or iid in rows:
                raise ValueError('empty or duplicate id')
            v = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(v).all():
                raise ValueError('non-finite prediction')
            if np.any((v[:256] < 0) | (v[:256] > 1)):
                raise ValueError('probability outside [0,1]')
            stats = v[256:].reshape(4, 2)
            if np.any(np.abs(stats[:, 0]) > 1) or np.any((stats[:, 1] < 0) | (stats[:, 1] > 1)):
                raise ValueError('invalid correlation or p-value')
            rows[iid] = v
    if not rows:
        raise ValueError('empty predictions')
    if expected_ids is not None and (len(set(expected_ids)) != len(expected_ids) or set(rows) != set(expected_ids)):
        raise ValueError('id set mismatch')
    return rows


def statistics(probabilities, obs_eval):
    """Author Eq. implementation: absolute DIFFERENCE IN TREATMENT EFFECTS.

    Preserve the code's b1 definition even though the paper prose calls it b1.
    First 2000 eligible OS rows for each signal; scipy two-sided Pearson p.
    """
    from scipy.stats import pearsonr
    models = np.asarray(probabilities).reshape(2, 4, 32)
    group = obs_eval[:, :5].astype(np.int64) @ (2**np.arange(4, -1, -1))
    s, a, y = obs_eval[:, 5:8].T
    bias = abs((models[0, 3] - models[0, 2]) - (models[1, 3] - models[1, 2]))[group]
    masks = {'S': np.ones(len(s), dtype=bool), 'A': s == 1,
             'Y0': (s == 1) & (a == 0), 'Y1': (s == 1) & (a == 1)}
    targets = {'S': s, 'A': a, 'Y0': y, 'Y1': y}
    result = []
    for key in SIGNALS:
        idx = np.flatnonzero(masks[key])[:2000]
        error = (targets[key][idx] - models[1, MODEL_KEYS.index(key), group[idx]])**2
        r, p = pearsonr(bias[idx], error)
        result.extend((float(r), float(p)))
    return np.asarray(result)
