"""CSV shape/identity/finite-value validation, independent of candidate code."""
import csv
import sys
import numpy as np

COLS=['id','w0','mu0','sd0','w1','mu1','sd1','n_components','amortised','forward_s','baseline_s']


def read_prediction(path):
    with open(path,newline='') as f:
        r=csv.reader(f)
        if next(r,None) != COLS:
            raise ValueError('incorrect CSV header')
        rows=list(r)
    if not rows or any(len(r)!=len(COLS) for r in rows):
        raise ValueError('empty file or wrong column count')
    ids=[r[0] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('duplicate ids')
    values=np.array([r[1:] for r in rows],dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('non-finite values')
    return ids,values


def validate(pred,inp):
    ids,v=read_prediction(pred)
    with np.load(inp,allow_pickle=False) as z:
        expected=z['id'].tolist()
        if z['phi'].shape != (len(expected),2) or z['z'].shape != (len(expected),1):
            raise ValueError('input shape mismatch')
    if len(ids)!=len(expected) or set(ids)!=set(expected):
        raise ValueError('id set mismatch')
    if (v[:,[0,2,3,5,8,9]]<=0).any():
        raise ValueError('weights, scales and measured times must be positive')
    if not np.allclose(v[:,0]+v[:,3],1,rtol=0,atol=2e-6):
        raise ValueError('weights do not sum to one')
    if not np.isin(v[:,6],[2]).all() or not (v[:,7]==1).all():
        raise ValueError('invalid component count or amortisation field')
    return len(ids)


if __name__=='__main__':
    try:
        print('ok:',validate(*sys.argv[1:]),'rows')
    except Exception as e:
        sys.exit(str(e))
