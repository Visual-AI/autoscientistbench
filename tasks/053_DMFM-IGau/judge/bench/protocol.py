"""Strict stream validation and the installed fossil 0.4.0 ARI semantics."""
import csv
import json
import math
from collections import Counter
HEADER = ["id", "iteration", "k", "M", "M_na"] + ["c"+str(i) for i in range(1,51)]
def load(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError("duplicate JSON key")
            out[k]=v
        return out
    def bad(v): raise ValueError("nonfinite JSON: "+v)
    with open(path) as f: return json.load(f, object_pairs_hook=pairs,parse_constant=bad)
def dimensions(meta):
    schema=meta.get("schema")
    if schema=="sta26-network-v1": count,iters,burn=50,100000,50000
    elif schema=="sta26-smoke-v1": count,iters,burn=2,200,100
    else: raise ValueError("unknown schema")
    ids=[x["id"] for x in meta["networks"]]
    if len(ids)!=count or len(set(ids))!=count: raise ValueError("invalid instance set")
    if any(not isinstance(x,str) or len(x)!=32 or any(c not in "0123456789abcdef" for c in x) for x in ids): raise ValueError("invalid opaque id")
    if "iterations" in meta and (meta["iterations"]!=iters or meta["burn_in"]!=burn): raise ValueError("label protocol mismatch")
    return ids,iters,burn
def rows(pred,meta):
    ids,iters,burn=dimensions(meta)
    with open(pred,newline="") as f:
        reader=csv.reader(f)
        if next(reader,None)!=HEADER: raise ValueError("wrong prediction header")
        for net_id in ids:
            for iteration in range(burn+1,iters+1):
                row=next(reader,None)
                if row is None or len(row)!=55: raise ValueError("missing/truncated/malformed draw")
                if row[0]!=net_id: raise ValueError("unknown, duplicate, reordered, or missing network")
                values=[]
                for text in row[1:]:
                    if not text or not text.isascii() or not text.isdecimal(): raise ValueError("non-integer/nonfinite draw field")
                    values.append(int(text))
                got,k,m,empty,*c=values
                if got!=iteration: raise ValueError("duplicate/missing/reordered/post-burn iteration")
                if not 1<=k<=m<=50 or empty!=m-k: raise ValueError("component guard violated")
                if set(c)!=set(range(1,k+1)): raise ValueError("noncanonical or inconsistent cluster labels")
                yield net_id,c,k
        if next(reader,None) is not None: raise ValueError("extra prediction rows")
def fossil_ari(c,truth,k=None):
    # Exact fossil::adj.rand.index: a=length(table(c)); a x a ctab;
    # absolute value; truth labels > a are excluded from the contingency,
    # while N remains length(c). Do not replace with signed sklearn ARI.
    k=k if k is not None else len(set(c))
    cells=Counter((x,y) for x,y in zip(c,truth) if 1<=x<=k and 1<=y<=k)
    columns=Counter(); rows_=Counter()
    for (x,y),v in cells.items(): columns[x]+=v; rows_[y]+=v
    choose=lambda n:n*(n-1)/2
    nij=sum(choose(v) for v in cells.values())
    ai=sum(choose(v) for v in columns.values()); bj=sum(choose(v) for v in rows_.values())
    expected=ai*bj/choose(len(c)); denom=0.5*(ai+bj)-expected
    if denom==0: raise ValueError("nonfinite fossil ARI")
    value=abs((nij-expected)/denom)
    if not math.isfinite(value) or not 0<=value<=1: raise ValueError("invalid fossil ARI")
    return value
