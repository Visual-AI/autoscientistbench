"""Strict stdlib parser shared by the frozen validator and scorer."""
import csv
import math
from pathlib import Path

FIELDS = ['id', 'exposure', 'outcome', 'mean', 'lower', 'upper']

def key(row):
    e, r = int(row['exposure']), int(row['outcome'])
    if not 1 <= e <= 9 or not 1 <= r <= 5:
        raise ValueError('invalid exposure/outcome index')
    return row['id'], e, r

def read_predictions(path):
    result = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('expected exactly ' + ','.join(FIELDS))
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('ragged prediction row')
            k = key(row)
            if k in result:
                raise ValueError('duplicate prediction key')
            values = tuple(float(row[c]) for c in ['mean', 'lower', 'upper'])
            if not all(math.isfinite(v) for v in values):
                raise ValueError('nonfinite prediction')
            if values[1] > values[2]:
                raise ValueError('reversed interval')
            result[k] = values
    if not result:
        raise ValueError('empty predictions')
    return result

def expected_keys(input_path):
    with open(Path(input_path) / 'manifest.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['id'] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('invalid observed manifest')
    return {(i,e,r) for i in ids for e in range(1,10) for r in range(1,6)}
