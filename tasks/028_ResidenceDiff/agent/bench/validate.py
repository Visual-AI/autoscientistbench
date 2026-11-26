"""Strict diffusion-design CSV validator."""
import csv
import sys
from pathlib import Path
import numpy as np


def read_design(pred_path, ids):
    ids=[str(i) for i in ids]
    if len(set(ids))!=len(ids): raise ValueError('duplicate input identifiers')
    seen={}
    with open(pred_path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','diffusivity']: raise ValueError('expected header id,diffusivity')
        for row in r:
            if None in row or any(v is None for v in row.values()): raise ValueError('wrong column count')
            i=row['id']
            if i in seen: raise ValueError('duplicate identifier')
            v=float(row['diffusivity'])
            if not np.isfinite(v) or v<=0: raise ValueError('diffusivity must be finite and positive')
            seen[i]=v
    if set(seen)!=set(ids): raise ValueError('identifier set mismatch')
    return np.array([seen[i] for i in ids])

if __name__=='__main__':
    try:
        with np.load(sys.argv[2],allow_pickle=False) as z: ids=z['id']
        read_design(sys.argv[1],ids)
    except Exception as e: sys.exit(str(e))
    print('ok')
