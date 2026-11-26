#!/usr/bin/env python3
"""Strict CSV validation. Usage: validate.py PRED INPUT."""
import csv, math, sys
import numpy as np
COLS=['id','m_deb','sigma_hat','asym_se']
def read_pred(path):
    rows={}
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=COLS:raise ValueError('header must be '+','.join(COLS))
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('wrong column count')
            iid=row['id']
            if not iid or iid in rows:raise ValueError('empty or duplicate id')
            vals=[float(row[k]) for k in COLS[1:]]
            if not all(math.isfinite(v) for v in vals):raise ValueError('nonfinite value')
            if vals[1]<=0 or vals[2]<=0:raise ValueError('standard error factors must be positive')
            if not math.isfinite(vals[1]*vals[2]):raise ValueError('standard error overflow')
            rows[iid]=vals
    if not rows:raise ValueError('empty prediction')
    return rows

def main(pred,inp):
    rows=read_pred(pred)
    with np.load(inp,allow_pickle=False) as z:ids=z['id'].astype(str).tolist()
    if len(ids)!=len(set(ids)) or set(rows)!=set(ids):raise ValueError('id set mismatch')
    print('ok: %d rows'%len(rows))
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
