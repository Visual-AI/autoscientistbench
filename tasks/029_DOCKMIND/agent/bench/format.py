"""Strict CSV schema shared by the trusted validator and scorer."""
import csv
import math
PARAMS = ('rho','exp_mean','exp_std','artifact_freq','artifact_mean','artifact_std')
BOUNDS = {'rho':(-.9,-.4),'exp_mean':(-4,4),'exp_std':(.5,3),'artifact_freq':(1e-7,1e-2),'artifact_mean':(-5,-3),'artifact_std':(.5,1)}
INPUT = ['id','run','target','pprop','definition']
PRED = ['id','pred',*PARAMS]
LEGACY_PRED = PRED
LABEL = [*INPUT,'hit_rate_mean']

def read(path, fields):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != fields: raise ValueError(f'expected header {fields}')
        rows=list(reader)
    ids=[]
    for r in rows:
        if None in r or any(v is None or v=='' for v in r.values()): raise ValueError('missing/extra field')
        ids.append(r['id'])
        for k in fields:
            if k not in ('id','target'):
                if not math.isfinite(float(r[k])): raise ValueError('nonfinite number')
        if 'run' in fields and r['run'] not in ('0','1','2','3','4'): raise ValueError('invalid run')
        if 'target' in fields and r['target'] not in ('ampc','d4','sigma2'): raise ValueError('invalid target')
    if not rows or len(set(ids)) != len(ids): raise ValueError('empty or duplicate ids')
    return rows

def aligned(pred, inputs):
    with open(pred,newline='') as stream:fields=next(csv.reader(stream),None)
    if fields != PRED:raise ValueError('six-parameter model certificate required')
    p=read(pred,fields)
    if [r['id'] for r in p] != [r['id'] for r in inputs]: raise ValueError('id set or order mismatch')
    groups={}
    for source,row in zip(inputs,p):
        params=tuple(float(row[key]) for key in PARAMS)
        for key,value in zip(PARAMS,params):
            low,high=BOUNDS[key]
            if not low<=value<=high:raise ValueError('parameter outside published bounds: '+key)
        group=(source['run'],source['target'])
        if group in groups and params!=groups[group]:raise ValueError('one parameter vector per target/run required')
        groups[group]=params
    return p
