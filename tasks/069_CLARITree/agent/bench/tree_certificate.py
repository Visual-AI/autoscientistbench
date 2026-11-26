"""Trusted tree checks and prediction; no submitted Python is imported here."""
import json
import math
from pathlib import Path
import numpy as np

MAX_CERTIFICATE_BYTES = 128 * 1024
N_FEATURES = 13

def threshold_pools(train_x):
    """Exact released global grid, including binary and low-cardinality cases."""
    pools = []
    for column in np.asarray(train_x, dtype=np.float64).T:
        if np.all((column == 0.) | (column == 1.)):
            pools.append([.5])
            continue
        values = sorted(map(float, column))
        unique = sorted(set(values))
        if len(unique) < 2:
            pools.append([])
        elif len(unique) <= 21:
            pools.append([(a+b)/2. for a,b in zip(unique[:-1], unique[1:])])
        else:
            thresholds = []
            for k in range(1, 21):
                position = (k/21.) * (len(values)-1)
                lower, upper = math.floor(position), math.ceil(position)
                thresholds.append(values[lower] + (position-lower)*(values[upper]-values[lower]))
            pools.append(sorted(set(thresholds)))
    return pools

def _number(value):
    return type(value) in (int, float) and math.isfinite(value)

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate certificate key')
        result[key] = value
    return result

def load_certificate(path):
    path = Path(path)
    if path.stat().st_size > MAX_CERTIFICATE_BYTES:
        raise ValueError('certificate too large')
    return json.loads(path.read_text(), object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def predict_checked(certificate, train_x, test_x, n_features=N_FEATURES):
    train_x, test_x = np.asarray(train_x), np.asarray(test_x)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != n_features or test_x.shape[1] != n_features:
        raise ValueError('feature shape')
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError('feature values')
    if not isinstance(certificate, dict) or set(certificate) != {'schema', 'tree'} or type(certificate['schema']) is not int or certificate['schema'] != 1:
        raise ValueError('certificate schema')
    pools = threshold_pools(train_x)
    # Same 0/1 test as threshold_pools: binary features are for splits only.
    binary = np.flatnonzero(np.all((train_x == 0.) | (train_x == 1.), axis=0))
    output = np.empty(len(test_x), dtype=np.float64)
    nodes = 0
    def walk(node, depth, train_rows, test_rows):
        nonlocal nodes
        nodes += 1
        if nodes > 31 or depth > 4 or not isinstance(node, dict):
            raise ValueError('tree structure')
        if set(node) == {'coefficients'}:
            coefficients = node['coefficients']
            if not isinstance(coefficients, list) or len(coefficients) != n_features+1 or not all(map(_number, coefficients)):
                raise ValueError('linear leaf coefficients')
            if any(coefficients[j+1] != 0 for j in binary):
                raise ValueError('coefficient on a binary column')
            with np.errstate(over='raise', invalid='raise'):
                output[test_rows] = coefficients[0] + test_x[test_rows] @ np.asarray(coefficients[1:], dtype=np.float64)
            return
        if set(node) != {'feature', 'threshold', 'left', 'right'}:
            raise ValueError('node keys')
        feature, threshold = node['feature'], node['threshold']
        if type(feature) is not int or not 0 <= feature < n_features or not _number(threshold):
            raise ValueError('split definition')
        if threshold not in pools[feature]:
            raise ValueError('threshold outside released grid')
        train_mask = train_x[train_rows, feature] <= threshold
        test_mask = test_x[test_rows, feature] <= threshold
        walk(node['left'], depth+1, train_rows[train_mask], test_rows[test_mask])
        walk(node['right'], depth+1, train_rows[~train_mask], test_rows[~test_mask])
    walk(certificate['tree'], 0, np.arange(len(train_x)), np.arange(len(test_x)))
    if not np.isfinite(output).all():
        raise ValueError('nonfinite prediction')
    return output

def r2(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != prediction.shape or y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError('target shape or values')
    with np.errstate(over='raise', invalid='raise'):
        total = np.sum((y-y.mean())**2)
        return 1.0 if total == 0 else float(1 - np.sum((y-prediction)**2)/total)
