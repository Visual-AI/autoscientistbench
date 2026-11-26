#!/usr/bin/env python3
"""Strict CSV layout validation. Row order is irrelevant; identity is mandatory."""
import csv
import sys
import numpy as np


def read_prediction(path, ids):
    expected=set(ids.tolist())
    with open(path,newline='') as f:
        r=csv.reader(f)
        if next(r,None)!=['id','x','y']: raise ValueError('header must be exactly id,x,y')
        rows={}
        for row in r:
            if len(row)!=3: raise ValueError('each row must have exactly three fields')
            i,x,y=row
            if i not in expected: raise ValueError('foreign ID')
            if i in rows: raise ValueError('duplicate ID')
            xy=(float(x),float(y))
            if not np.isfinite(xy).all(): raise ValueError('coordinates must be finite')
            rows[i]=xy
    if set(rows)!=expected: raise ValueError('missing IDs')
    return np.asarray([rows[i] for i in ids],dtype=np.float64)


def main(pred,input_path):
    with np.load(input_path,allow_pickle=False) as z: ids=z['ids']
    coords=read_prediction(pred,ids)
    if coords.shape!=(len(ids),2): raise ValueError('wrong shape')
    print('ok')

if __name__=='__main__':
    try: main(*sys.argv[1:])
    except Exception as e:
        print(str(e),file=sys.stderr); sys.exit(1)
