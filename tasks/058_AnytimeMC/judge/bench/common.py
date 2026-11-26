"""Strict formats shared by the frozen validator and scorer; Python stdlib only."""
import csv,json,math
from pathlib import Path
FIELDS=["trial_id","hypothesis","rejected","draws","sampler_only"]
INPUT_FIELDS=["trial_id","hypothesis","observed","B","alpha"]

def load_json(path):
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d: raise ValueError("duplicate JSON key")
            d[k]=v
        return d
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def integer(x):
    if not isinstance(x,str) or not x.isascii() or not x.isdecimal(): raise ValueError("expected nonnegative integer")
    return int(x)

def rows(path,fields):
    with open(path,newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=fields: raise ValueError("wrong CSV header")
        data=list(reader)
    if not data or any(set(r)!=set(fields) or any(v is None for v in r.values()) for r in data): raise ValueError("empty or malformed CSV")
    return data

def inputs(path):
    groups={}
    for r in rows(path,INPUT_FIELDS):
        ident=r["trial_id"]
        if len(ident)!=32 or any(c not in "0123456789abcdef" for c in ident): raise ValueError("bad trial ID")
        h=integer(r["hypothesis"]);B=integer(r["B"]);z=float(r["observed"]);alpha=float(r["alpha"])
        if not math.isfinite(z) or B<1 or B>10000 or alpha!=0.1: raise ValueError("invalid observation/protocol")
        g=groups.setdefault(ident,{"M":0,"B":B,"alpha":alpha})
        if g["B"]!=B or h!=g["M"]+1: raise ValueError("missing/duplicate hypothesis")
        g["M"]+=1
    return groups

def labels(path):
    y=load_json(path)
    if set(y)!={"version","scope","protocol","input_sha256","trials"} or y["version"]!=1: raise ValueError("bad labels")
    scope=y["scope"]; p=y["protocol"]
    dims=(1000,10000,10) if scope=="full" else (24,256,2) if scope=="smoke" else None
    if dims is None: raise ValueError("unknown scope")
    expected=dict(zip(("M","B","trials"),dims),pi_A=0.4,mu_A=2.5,alpha=0.1,h=10,b=0.9)
    if p!=expected or len(y["trials"])!=p["trials"]: raise ValueError("incomplete protocol")
    groups={};truth={}
    for t in y["trials"]:
        if set(t)!={"trial_id","alternative"}: raise ValueError("invalid truth row")
        ident=t["trial_id"];a=t["alternative"]
        if ident in groups or len(a)!=p["M"] or any(type(x)!=int or x not in (0,1) for x in a): raise ValueError("invalid truth")
        groups[ident]={"M":p["M"],"B":p["B"],"alpha":p["alpha"]};truth[ident]=a
    return y,groups,truth

def predictions(path,groups):
    out={i:{} for i in groups}
    for r in rows(path,FIELDS):
        ident=r["trial_id"]
        if ident not in groups: raise ValueError("unknown trial")
        h=integer(r["hypothesis"]);rej=integer(r["rejected"]);draws=integer(r["draws"]);sampler=integer(r["sampler_only"])
        if h<1 or h>groups[ident]["M"] or h in out[ident]: raise ValueError("duplicate/out-of-range hypothesis")
        if rej not in (0,1) or sampler not in (0,1) or not 1<=draws<=groups[ident]["B"]: raise ValueError("invalid decision/count/interface flag")
        out[ident][h]=(rej,draws,sampler)
    if any(len(out[i])!=g["M"] for i,g in groups.items()): raise ValueError("missing predictions")
    return out
