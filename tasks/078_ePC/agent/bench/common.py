"""Frozen CSV prediction contract shared by validator, scorer, and certificate probe."""
import csv
import numpy as np
HEADER = ['id'] + [f'pred{s}' for s in range(5)]

def read_predictions(path):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        if next(reader, None) != HEADER:
            raise ValueError('header must be ' + ','.join(HEADER))
        ids, values = [], []
        for row in reader:
            if len(row) != 6:
                raise ValueError('expected six columns')
            i = int(row[0])
            if str(i) != row[0] or i < 0:
                raise ValueError('non-canonical id')
            v = np.asarray([float(x) for x in row[1:]])
            if not np.all(np.isfinite(v)) or not np.all((v >= 0) & (v <= 9) & (v == np.floor(v))):
                raise ValueError('each prediction must be an integer class in 0..9')
            ids.append(i); values.append(v.astype(np.int64))
    if not ids or len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError('ids must be nonempty, unique, and sorted')
    return np.asarray(ids, dtype=np.int64), np.stack(values)
