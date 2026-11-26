#!/usr/bin/env python3
"""Strict CSV schema, finite values, and exact input/output id alignment."""
import csv,math,sys
COLS=['id','pred']
LEGACY_COLS=['id','pred','cpu_s','analytic']
def read_pred(path):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames not in (COLS,LEGACY_COLS):raise ValueError('header must be id,pred or id,pred,cpu_s,analytic')
        columns=reader.fieldnames
        rows=list(reader)
    if not rows:raise ValueError('empty predictions')
    ids=set();out=[]
    for r in rows:
        if None in r or any(v is None for v in r.values()):raise ValueError('wrong column count')
        if not r['id'] or r['id'] in ids:raise ValueError('empty or duplicate id')
        ids.add(r['id'])
        for key in columns[1:]:
            r[key]=float(r[key])
            if not math.isfinite(r[key]):raise ValueError('nonfinite '+key)
        if not -1e-8<=r['pred']<=1+1e-8:raise ValueError('concentration outside [0,1]')
        if columns==LEGACY_COLS and (r['cpu_s']<0 or r['analytic'] not in (0.,1.)):raise ValueError('invalid diagnostic')
        out.append(r)
    if columns==LEGACY_COLS and (len({r['cpu_s'] for r in out})!=1 or len({r['analytic'] for r in out})!=1):raise ValueError('curve diagnostics must agree')
    return out

def main(pred,inp):
    rows=read_pred(pred)
    with open(inp,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','p']:raise ValueError('invalid input schema')
        ids=[v['id'] for v in r]
    if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in rows}:raise ValueError('id sets differ')
    print(f'ok: {len(rows)} rows')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
