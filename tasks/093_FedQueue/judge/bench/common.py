"""Frozen prediction parser shared by validation, scoring and the probe."""
import csv
import numpy as np

HEADER=['id','pred','total_local_steps','rounds','clients','protocol','selected_round']
PROTOCOL='fedqueue-alpha01-rho09-v1'

def read_predictions(path):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=HEADER:raise ValueError('incorrect prediction header')
        rows=list(reader)
    if not rows:raise ValueError('empty prediction')
    parsed=[]
    for row in rows:
        if set(row)!=set(HEADER) or any(row[k] is None for k in HEADER):raise ValueError('malformed row')
        r={k:(row[k] if k=='protocol' else int(row[k])) for k in HEADER}
        if not 0<=r['pred']<=9:raise ValueError('pred must be an integer class in 0..9')
        if r['id']<0 or r['total_local_steps']<0:raise ValueError('negative id or work')
        parsed.append(r)
    ids=np.array([r['id'] for r in parsed],dtype=np.int64)
    if np.any(np.diff(ids)<=0):raise ValueError('ids must be unique and sorted')
    for k in HEADER[2:]:
        if any(r[k]!=parsed[0][k] for r in parsed):raise ValueError('inconsistent run metadata')
    return ids,np.array([r['pred'] for r in parsed],dtype=np.int64),{k:parsed[0][k] for k in HEADER[2:]}

def check_ids(ids,expected):
    if not np.array_equal(ids,expected):raise ValueError('prediction ids differ from input/label ids')
