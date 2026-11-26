"""Validate the complete CSV shape; physical guards are checked by scorer."""
import csv
import json
import math
import sys
import numpy as np

COLUMNS=['id','replica','permutation','elapsed_s','iterations']
REPLICAS=10

def read_predictions(pred_path,input_path):
    with np.load(input_path,allow_pickle=False) as z:
        ids=[str(x) for x in z['id']];sizes=dict(zip(ids,map(int,z['n'])))
    if len(set(ids))!=len(ids):raise ValueError('duplicate input ID')
    expected={(iid,r) for iid in ids for r in range(REPLICAS)}
    rows={}
    with open(pred_path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=COLUMNS:raise ValueError('incorrect CSV header')
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('incorrect column count')
            iid=row['id'];rep=int(row['replica']);key=(iid,rep)
            if key not in expected:raise ValueError('unknown id or replica')
            if key in rows:raise ValueError('duplicate id/replica')
            perm=json.loads(row['permutation'])
            if not isinstance(perm,list) or len(perm)!=sizes[iid]:raise ValueError('incorrect permutation shape')
            if any(type(x) is not int for x in perm):raise ValueError('indices must be integers')
            elapsed=float(row['elapsed_s']);iterations=int(row['iterations'])
            if not math.isfinite(elapsed) or elapsed<0 or iterations<0:raise ValueError('invalid counters')
            rows[key]=dict(permutation=perm,elapsed_s=elapsed,iterations=iterations)
    if set(rows)!=expected:raise ValueError('missing id/replica rows')
    return rows

def main():
    try:rows=read_predictions(sys.argv[1],sys.argv[2])
    except Exception as e:sys.exit(str(e))
    print(f'ok: {len(rows)} rows')
if __name__=='__main__':main()
