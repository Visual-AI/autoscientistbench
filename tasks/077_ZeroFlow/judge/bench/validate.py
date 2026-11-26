"""Strict CSV matrix format check; arbitrary instance row order is permitted."""
import csv,sys
from pathlib import Path
import numpy as np

HEADER=['id']+['g%d_%d'%(i,j) for i in range(50) for j in range(50)]

def read_prediction(path,ids):
    ids=[str(i) for i in ids]
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('invalid reference IDs')
    pred={}
    with open(path,newline='') as f:
        r=csv.reader(f)
        if next(r,None)!=HEADER:raise ValueError('wrong header')
        for row in r:
            if len(row)!=2501:raise ValueError('wrong row width')
            ident=row[0]
            if ident in pred:raise ValueError('duplicate id')
            v=np.array(row[1:],dtype=np.float64)
            if not np.isfinite(v).all():raise ValueError('non-finite gate score')
            pred[ident]=v.reshape(50,50)
    if set(pred)!=set(ids):raise ValueError('id mismatch')
    return np.stack([pred[i] for i in ids])

if __name__=='__main__':
    try:
        with np.load(sys.argv[2],allow_pickle=False) as z:
            if set(z.files)!={'id','x','trial'}:raise ValueError('wrong input keys')
            ids=z['id'];x=z['x'];trial=z['trial']
            if ids.ndim!=1 or x.shape!=(len(ids),2048,50) or trial.shape!=(len(ids),):raise ValueError('wrong input shape')
            if not np.isfinite(x).all():raise ValueError('non-finite observations')
        read_prediction(sys.argv[1],ids)
    except Exception as e:
        print(str(e),file=sys.stderr);sys.exit(1)
    print('ok')
