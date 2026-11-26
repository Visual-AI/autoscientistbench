#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import read_predictions

def validate(pred, inp):
    ids, values = read_predictions(pred)
    with np.load(inp, allow_pickle=False) as z:
        if set(z.files) != {'id', 'x'}:
            raise ValueError('input keys must be id,x')
        expected, x = z['id'], z['x']
        if expected.ndim != 1 or expected.dtype.kind not in 'iu' or len(np.unique(expected)) != len(expected):
            raise ValueError('invalid input ids')
        if x.dtype != np.uint8 or x.shape != (len(expected), 28, 28):
            raise ValueError('input shape/dtype mismatch')
        if not np.array_equal(ids, expected):
            raise ValueError('prediction ids must equal input ids in order')
    return len(ids)
if __name__ == '__main__':
    try: print(f'ok: {validate(sys.argv[1], sys.argv[2])} rows')
    except Exception as e: sys.exit(str(e))
