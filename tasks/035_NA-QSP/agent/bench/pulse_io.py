"""Strict, text-only pulse submission format. No physics is trusted from the submission."""
import csv
import json
import math


def instances(path):
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames != ['id','alpha','theta_t']:raise ValueError('bad instance header')
        rows=list(r)
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('empty/duplicate instance ids')
    for r in rows:
        if set(r)!= {'id','alpha','theta_t'}:raise ValueError('bad instance columns')
        for key in ('alpha','theta_t'):
            r[key]=float(r[key])
            if not math.isfinite(r[key]) or r[key]<=0:raise ValueError('bad instance parameter')
    return rows


def no_duplicate_keys(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate gate key')
        d[k]=v
    return d


def check_finite(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k != 'type':check_finite(v)
    elif isinstance(value,list):
        for v in value:check_finite(v)
    elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('nonfinite or nonnumeric pulse parameter')


def predictions(path, ids):
    out={}
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','pulse','n_gates']:raise ValueError('header must be id,pulse,n_gates')
        for row in r:
            if set(row)!= {'id','pulse','n_gates'}:raise ValueError('extra columns')
            key=row['id']
            if key in out:raise ValueError('duplicate id')
            gates=json.loads(row['pulse'],object_pairs_hook=no_duplicate_keys)
            if not isinstance(gates,list) or any(not isinstance(g,dict) for g in gates):raise ValueError('pulse must be a list of gate objects')
            if str(len(gates))!=row['n_gates']:raise ValueError('n_gates must match pulse length')
            check_finite(gates)
            out[key]=gates
    if set(out)!=set(ids):raise ValueError('prediction id set mismatch')
    return out
