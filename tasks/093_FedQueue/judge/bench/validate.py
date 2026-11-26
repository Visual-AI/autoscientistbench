#!/usr/bin/env python3
"""Usage: validate.py PRED INPUT; rejects malformed, incomplete or reordered output."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_predictions,check_ids

def main(pred,input_path):
    ids,_,_=read_predictions(pred)
    with np.load(input_path,allow_pickle=False) as z:
        if set(z.files)!={'id','x'}:raise ValueError('input keys must be id,x')
        expected=z['id'];x=z['x']
        if expected.ndim!=1 or expected.dtype.kind not in 'iu':raise ValueError('invalid ids')
        if x.shape!=(len(expected),28,28) or x.dtype!=np.uint8:raise ValueError('invalid image shape/dtype')
        if np.any(np.diff(expected)<=0):raise ValueError('input ids not sorted/unique')
        check_ids(ids,expected)
    print(f'ok: {len(ids)} rows')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:print(f'invalid: {e}',file=sys.stderr);sys.exit(1)
