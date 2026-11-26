"""Frozen CSV contracts shared by validation and scoring."""
import csv
import json
import math


def read_table(path, columns, numeric):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=columns:raise ValueError(f'expected header {columns}')
        rows=list(reader)
    if not rows:raise ValueError('empty table')
    ids=[];values=[]
    for row in rows:
        if None in row or any(v is None for v in row.values()):raise ValueError('malformed row')
        i=row['id']
        if not i.isdecimal() or str(int(i))!=i:raise ValueError('id must be canonical nonnegative integer')
        ids.append(i)
        if numeric:
            x=float(row[numeric])
            if not math.isfinite(x) or x<0:raise ValueError('minutes must be finite and nonnegative')
            values.append(x)
    if len(set(ids))!=len(ids):raise ValueError('duplicate id')
    if ids!=sorted(ids,key=int):raise ValueError('rows must be sorted by numeric id')
    return ids,values


def input_ids(path):
    ids,_=read_table(path,['id','responses'],None)
    return ids
