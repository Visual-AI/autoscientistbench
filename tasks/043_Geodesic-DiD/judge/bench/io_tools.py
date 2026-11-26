import csv
import math
from pathlib import Path

HEADER = ['id'] + [f'{kind}_{i:03d}' for kind in ('start', 'end') for i in range(101)]


def read_predictions(pred, expected):
    with open(pred, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError('prediction columns must be id, start_000..100, end_000..100')
        rows = list(reader)
    if len(rows) != len(expected) or [r['id'] for r in rows] != expected:
        raise ValueError('one prediction per input id in input order is required')
    for row in rows:
        values = [float(row[k]) for k in HEADER[1:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('non-finite prediction')
        for q in (values[:101], values[101:]):
            if abs(q[0] + 3) > 1e-7 or abs(q[-1] - 3) > 1e-7:
                raise ValueError('quantile support endpoints must be -3 and 3')
            if any(b <= a for a, b in zip(q, q[1:])):
                raise ValueError('prediction must define a strictly increasing quantile curve')
    return rows


def input_ids(path):
    with open(Path(path) / 'index.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['id'] for r in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('invalid input ids')
    return ids
