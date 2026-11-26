"""Strict CSV schema and observation-id coverage check."""
import csv
import sys
from pathlib import Path
import numpy as np

HEADER=['id','t','f1','f2','f3','f4','f5']


def read_prediction(path, ids):
    ids=[str(i) for i in ids]
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('input ids must be nonempty and unique')
    arrays={i:np.empty((200,5),dtype=np.float64) for i in ids}
    seen=set()
    with open(path,newline='') as f:
        r=csv.reader(f)
        if next(r,None)!=HEADER:raise ValueError('incorrect header')
        for row in r:
            if len(row)!=7:raise ValueError('incorrect column count')
            iid=row[0]
            if iid not in arrays:raise ValueError('unknown id')
            t=int(row[1])
            if str(t)!=row[1] or not 0<=t<200:raise ValueError('invalid time index')
            if (iid,t) in seen:raise ValueError('duplicate id/time')
            v=np.asarray(row[2:],dtype=np.float64)
            if not np.isfinite(v).all():raise ValueError('nonfinite factor')
            arrays[iid][t]=v
            seen.add((iid,t))
    if len(seen)!=200*len(ids):raise ValueError('missing rows: require (200,5) per instance')
    return np.stack([arrays[i] for i in ids])

if __name__=='__main__':
    try:
        with np.load(sys.argv[2],allow_pickle=False) as z:ids=z['id']
        p=read_prediction(sys.argv[1],ids)
        print(f'ok: {len(p)} complete factor matrices')
    except Exception as e:
        print(str(e),file=sys.stderr);sys.exit(1)
