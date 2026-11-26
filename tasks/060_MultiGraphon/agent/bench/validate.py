#!/usr/bin/env python3
"""validate.py PRED INPUT: exact ids, tensor shape, probability and timing checks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import read_npz, read_prediction

if __name__ == '__main__':
    try:
        obs = read_npz(sys.argv[2])
        pred = read_prediction(sys.argv[1], obs['id'].tolist())
        print('ok:', len(pred), 'collections')
    except Exception as e:
        sys.exit(str(e))
