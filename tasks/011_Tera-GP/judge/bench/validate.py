#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from formats import predictions

def main(pred,input_):
    rows,_=predictions(pred); expected=[]
    with np.load(input_,allow_pickle=False) as z:
        keys={f'd{d}_r{r}_{suffix}' for d in range(6) for r in range(3) for suffix in ['id','R']}
        if set(z.files)!=keys: raise ValueError('Input keys differ from the specification')
        for d,atoms in enumerate([56,60,87,118,148,370]):
            for r in range(3):
                ids=z[f'd{d}_r{r}_id']; xyz=z[f'd{d}_r{r}_R']
                if ids.ndim!=1 or xyz.shape!=(len(ids),atoms,3): raise ValueError('Input shape mismatch')
                if not np.isfinite(xyz).all(): raise ValueError('Non-finite coordinates')
                expected.extend(ids.tolist())
    if len(expected)!=len(set(expected)):raise ValueError('Duplicate input ID')
    if set(rows)!=set(expected):raise ValueError('Prediction ID set differs from input')
    print('ok:',len(rows),'rows')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
