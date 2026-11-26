"""Frozen CSV parsing and alignment shared by validator and scorer."""
import csv
import math

HEADER=['id','pred','train_s','peak_gb','cpus']

def predictions(path):
    rows={}; certificate=None
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=HEADER: raise ValueError('Prediction header must be '+','.join(HEADER))
        for row in r:
            if None in row or any(v is None for v in row.values()): raise ValueError('Ragged CSV')
            id_=row['id']
            if not id_ or id_ in rows: raise ValueError('Empty or duplicate ID')
            values=[float(row[k]) for k in HEADER[1:]]
            if not all(math.isfinite(v) for v in values): raise ValueError('Non-finite number')
            if min(values[1:])<0: raise ValueError('Negative resource value')
            current=tuple(values[1:])
            if certificate is not None and current!=certificate: raise ValueError('Resource certificate differs between rows')
            certificate=current; rows[id_]=values[0]
    if not rows: raise ValueError('Empty predictions')
    return rows,certificate
