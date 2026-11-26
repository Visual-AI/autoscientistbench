"""Validate a finite CI table against an observation collection."""
import csv
import math
import sys
import numpy as np

COLUMNS = ['id', 'point', 'lower', 'upper']


def read_prediction(path):
    result={}
    with open(path, newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError('header must be id,point,lower,upper')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('wrong number of columns')
            iid=row['id']
            if not iid or iid in result:
                raise ValueError('empty or duplicate id')
            point,lo,hi=(float(row[k]) for k in COLUMNS[1:])
            if not all(math.isfinite(v) for v in (point,lo,hi)):
                raise ValueError('nonfinite prediction')
            if lo>hi:
                raise ValueError('reversed interval')
            result[iid]=(point,lo,hi)
    if not result: raise ValueError('empty prediction')
    return result


def validate(pred,inp):
    values=read_prediction(pred)
    with np.load(inp,allow_pickle=False) as z:
        ids=[str(x) for x in z['id']]
    if len(ids)!=len(set(ids)) or set(values)!=set(ids):
        raise ValueError('id set mismatch')
    return len(ids)


if __name__=='__main__':
    try: print('ok:',validate(*sys.argv[1:]),'rows')
    except Exception as e: sys.exit(str(e))
