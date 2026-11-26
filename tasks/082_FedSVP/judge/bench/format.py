"""Frozen CSV schema validation, shared by scorer and validator."""
import csv
import numpy as np
HEADER=['id','global']+[f'personal_{c}' for c in range(10)]
def read_predictions(path):
    with open(path,newline='') as f:
        rows=list(csv.reader(f))
    if not rows or rows[0]!=HEADER or len(rows)<2:raise ValueError('bad or empty CSV/header')
    if any(len(r)!=12 for r in rows[1:]):raise ValueError('column count')
    values=np.array([[int(v) for v in r] for r in rows[1:]],dtype=np.int64)
    ids=values[:,0];pred=values[:,1:]
    if np.any(ids<0) or np.any(np.diff(ids)<=0):raise ValueError('IDs must be unique and sorted')
    if np.any(pred<0) or np.any(pred>=101):raise ValueError('class outside 0..100')
    return ids,pred
