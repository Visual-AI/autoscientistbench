#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction import read_predictions

def main(pred, inp):
    rows, metadata = read_predictions(pred)
    with np.load(inp, allow_pickle=False) as z: ids = z['id'].tolist()
    if len(ids) != len(set(ids)) or set(rows) != set(ids): raise ValueError('id set mismatch')
    print(f'ok: {len(rows)} rows')

if __name__ == '__main__':
    try: main(*sys.argv[1:])
    except Exception as e: sys.exit(str(e))
