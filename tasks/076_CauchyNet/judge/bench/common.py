"""Frozen CSV schema shared by validation and scoring."""
import csv, math
import numpy as np

HEADER=['id']+[f'pred_{s}' for s in range(10)]+[f'params_{s}' for s in range(10)]

def read_csv(path, header):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != header: raise ValueError('incorrect header')
        rows=list(reader)
    ids=[r['id'] for r in rows]
    if not ids or any(not i for i in ids) or len(ids)!=len(set(ids)): raise ValueError('empty or duplicate ids')
    if any(None in r or any(v is None for v in r.values()) for r in rows):raise ValueError('incorrect row width')
    return rows

def predictions(path):
    rows=read_csv(path,HEADER)
    p=np.asarray([[float(r[f'pred_{s}']) for s in range(10)] for r in rows],dtype=np.float32)
    c=np.asarray([[float(r[f'params_{s}']) for s in range(10)] for r in rows],dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(c).all():raise ValueError('non-finite values')
    if (c<0).any() or (c!=np.floor(c)).any():raise ValueError('parameter count must be a nonnegative integer')
    if not (c==c[0]).all():raise ValueError('parameter count varies between rows')
    return [r['id'] for r in rows],p,c[0]
