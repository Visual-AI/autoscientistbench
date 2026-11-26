"""Validate one binary call per opaque input ID; no pickle loading."""
import csv
import sys
import numpy as np


def read_predictions(path):
    result = {}
    with open(path, newline='') as f:
        r = csv.reader(f)
        if next(r, None) != ['id', 'called']:
            raise ValueError('header must be id,called')
        for row in r:
            if len(row) != 2 or not row[0] or row[1] not in ('0', '1'):
                raise ValueError('rows require nonempty ID and finite binary called (0 or 1)')
            if row[0] in result:
                raise ValueError('duplicate ID')
            result[row[0]] = int(row[1])
    if not result:
        raise ValueError('empty prediction')
    return result


def main(pred, inp):
    p = read_predictions(pred)
    with np.load(inp, allow_pickle=False) as z:
        ids = z['id'].tolist()
    if len(ids) != len(set(ids)) or set(p) != set(ids):
        raise ValueError('prediction/input ID sets differ or input IDs duplicate')
    print(f'ok: {len(p)} binary calls')


if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except Exception as e:
        sys.exit(str(e))
