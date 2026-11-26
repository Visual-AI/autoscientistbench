"""Frozen standard-library schema checks shared by validator and scorer."""
import json
import math

def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k] = v
        return out
    def constant(value):
        raise ValueError("nonfinite JSON number")
    with open(path) as f:
        return json.load(f, object_pairs_hook=pairs, parse_constant=constant)

def predictions(obj, ids):
    if not isinstance(obj,dict) or set(obj)!={"version","predictions"} or type(obj['version']) is not int or obj['version']!=1:
        raise ValueError("prediction envelope")
    rows = obj['predictions']
    if not isinstance(rows,list) or len(rows)!=len(ids):
        raise ValueError("prediction count")
    out={}
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"id","p_kn"} or not isinstance(row['id'],str) or row['id'] not in ids or row['id'] in out:
            raise ValueError("unknown or duplicate ID")
        p=row['p_kn']
        if not isinstance(p,list) or len(p)!=500 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in p):
            raise ValueError("500 finite probabilities required")
        if abs(math.fsum(p)-1)>1e-8:
            raise ValueError("posterior probabilities must sum to one")
        out[row['id']]=p
    if set(out)!=set(ids):
        raise ValueError("missing IDs")
    return out
