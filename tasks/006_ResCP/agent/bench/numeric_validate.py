"""Validate CSV intervals against the ids of a supplied history archive."""
import csv
import sys
from pathlib import Path
import numpy as np

def read_prediction(path):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','lower','upper']:raise ValueError('header must be id,lower,upper')
        rows=list(reader)
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate id')
    values=np.array([[float(r['lower']),float(r['upper'])] for r in rows])
    if values.shape!=(len(ids),2) or not np.isfinite(values).all():raise ValueError('nonfinite or malformed intervals')
    if np.any(values[:,0]>values[:,1]):raise ValueError('lower exceeds upper')
    return ids,values

def main(pred,input_path):
    a=np.load(input_path,allow_pickle=False)
    expected=np.concatenate([a[k].reshape(-1) for k in sorted(a.files) if k.endswith('__id')]).tolist()
    ids,_=read_prediction(pred)
    if set(ids)!=set(expected) or len(ids)!=len(expected):raise ValueError('id set mismatch')
    print('ok:',len(ids),'intervals')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
