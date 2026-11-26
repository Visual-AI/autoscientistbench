#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from common import read_predictions

def validate(pred_path,input_path):
    ids,_,_,_=read_predictions(pred_path)
    with np.load(input_path,allow_pickle=False) as a:
        if set(a.files)!={'id','X'}:raise ValueError('input keys must be id,X')
        expected=a['id'];X=a['X']
        if expected.ndim!=1 or expected.dtype.kind not in 'iu' or len(set(expected.tolist()))!=len(expected):raise ValueError('invalid input ids')
        if X.shape!=(len(expected),617) or not np.isfinite(X).all():raise ValueError('invalid input features')
        if not np.array_equal(ids,expected):raise ValueError('id set/order mismatch')
    return len(ids)
if __name__=='__main__':
    try:print(f'ok: {validate(*sys.argv[1:])} rows')
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
