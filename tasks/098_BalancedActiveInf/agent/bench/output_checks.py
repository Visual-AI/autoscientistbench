"""Strict output schema and auditable per-replication sampling budgets."""
import json
import math
import numpy as np


def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def check(pred_path,meta):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d: raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    with open(pred_path) as f: out=json.load(f,object_pairs_hook=unique)
    if set(out)!={'version','predictions_sha256','designs','trials'} or out['version']!=1:
        raise ValueError('wrong output schema')
    if out['predictions_sha256']!=meta['predictions_sha256']: raise ValueError('frozen prediction identity mismatch')
    N=meta['population_size']; T=meta['n_replications']
    if meta['scope']=='full' and T!=10000: raise ValueError('full trial count changed')
    designs=out['designs']; records=out['trials']
    if not isinstance(designs,dict) or not designs or len(designs)>T: raise ValueError('invalid designs')
    budgets={}
    for name,d in designs.items():
        if set(d)!={'dtype','pi'} or d['dtype'] not in ['float32','float64']: raise ValueError('invalid design dtype')
        if not isinstance(d['pi'],list) or len(d['pi'])!=N or not all(number(p) and 0<p<=1 for p in d['pi']):
            raise ValueError('invalid inclusion probabilities')
        p=np.asarray(d['pi'],dtype=d['dtype'])
        if not np.isfinite(p).all() or np.any(p<=0): raise ValueError('invalid rounded probabilities')
        # Original evaluator uses NumPy sum in the design array's dtype.
        budgets[name]=float(p.sum())
    if not isinstance(records,list) or len(records)!=T: raise ValueError('wrong replication count')
    seen=set()
    for row in records:
        if set(row)!={'id','design','selected','lower','upper','width'}: raise ValueError('wrong trial schema')
        i=row['id']
        if type(i) is not int or i<0 or i>=T or i in seen: raise ValueError('invalid/duplicate trial id')
        seen.add(i)
        if row['design'] not in designs: raise ValueError('unknown design')
        ids=row['selected']
        if not isinstance(ids,list) or any(type(i) is not int or i<0 or i>=N for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('invalid/duplicate sampled unit')
        lo,hi,width=(row[k] for k in ['lower','upper','width'])
        if not all(number(x) for x in [lo,hi,width]) or width<0 or hi<lo: raise ValueError('invalid interval')
        if not math.isclose(hi-lo,width,rel_tol=1e-12,abs_tol=1e-10): raise ValueError('inconsistent width')
    return out,budgets
