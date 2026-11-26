"""Strict CSV schema and hypothesis-ID validation."""
import csv,sys
from pathlib import Path
import numpy as np
COLS=['id','beta','var','p_value']

def read_predictions(path,ids):
    expected=list(map(str,ids));assert len(expected)==len(set(expected)), 'duplicate input IDs'
    seen={}
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=COLS:raise ValueError('header must be '+','.join(COLS))
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('wrong column count')
            iid=row['id']
            if iid in seen:raise ValueError('duplicate ID')
            values=np.array([float(row[k]) for k in COLS[1:]])
            if not np.all(np.isfinite(values)):raise ValueError('nonfinite prediction')
            if values[1]<=0:raise ValueError('variance must be positive')
            if not 0<=values[2]<=1:raise ValueError('p_value outside [0,1]')
            seen[iid]=values
    if set(seen)!=set(expected):raise ValueError('prediction ID set mismatch')
    return np.stack([seen[k] for k in expected])

def main(pred,inp):
    with np.load(inp,allow_pickle=False) as z:ids=z['hypothesis_id'].ravel()
    values=read_predictions(pred,ids)
    print('ok: '+str(len(values))+' hypotheses')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
