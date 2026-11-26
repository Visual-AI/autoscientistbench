#!/usr/bin/env python3
"""Usage: validate.py PRED INPUT; reject invalid CSV, IDs, ranges and shapes."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import read_prediction

if __name__ == '__main__':
    try:
        with np.load(sys.argv[2], allow_pickle=False) as z:
            ids = z['id'].astype(str).tolist()
        rows = read_prediction(sys.argv[1], ids)
        print(f'ok: {len(rows)} instances')
    except Exception as e:
        sys.exit(f'invalid: {e}')
