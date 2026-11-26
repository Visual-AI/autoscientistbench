"""Frozen, replayable predictable betting policies.

A single data-independent certificate is shared by every row. Expressions only
see strictly past sufficient statistics, never the current observation or U.
"""
import hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from betting_core import betting_mart,lambda_LBOW,lambda_predmix_eb

DEFAULT = {'version':1,'B':100,'positive':'lbow','negative':'predmix','theta':0.5,'trunc_scale':0.5,'combine':'max'}
VARIABLES={'lbow','predmix','t','n','mean','variance','sum','sumsq'}
OPS={'add':2,'sub':2,'mul':2,'div':2,'min':2,'max':2,'abs':1,'sqrt':1,'log':1,'neg':1}

def check_expression(e,depth=0):
    if depth>12:raise ValueError('expression too deep')
    if type(e) in (int,float):
        if not math.isfinite(e) or abs(e)>1e6:raise ValueError('invalid constant')
    elif type(e)==str:
        if e not in VARIABLES:raise ValueError('unknown predictable feature')
    elif type(e)==list and len(e)>0 and type(e[0])==str and e[0] in OPS:
        if len(e)!=OPS[e[0]]+1:raise ValueError('operator arity')
        for a in e[1:]:check_expression(a,depth+1)
    else:raise ValueError('invalid expression')

def check_policy(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):raise ValueError('certificate keys')
    if type(p['version'])!=int or p['version']!=1:raise ValueError('certificate version')
    if type(p['B'])!=int or not 1<=p['B']<=1000:raise ValueError('B outside 1..1000')
    for k in ('theta','trunc_scale'):
        if type(p[k]) not in (float,int) or not math.isfinite(p[k]):raise ValueError(k)
    if not 0<=p['theta']<=1 or not 0<p['trunc_scale']<=1:raise ValueError('capital bounds')
    if p['combine'] not in ('max','mixture'):raise ValueError('capital combination')
    check_expression(p['positive']);check_expression(p['negative'])
    if len(json.dumps(p))>8192:raise ValueError('certificate too long')
    return p

def canonical(p):
    check_policy(p)
    return json.dumps(p,sort_keys=True,separators=(',',':'))

def _eval(e,features):
    if type(e) in (int,float):return e
    if type(e)==str:return features[e]
    args=[_eval(a,features) for a in e[1:]]
    op=e[0]
    if op=='add':return args[0]+args[1]
    if op=='sub':return args[0]-args[1]
    if op=='mul':return args[0]*args[1]
    if op=='div':return np.divide(args[0],args[1])
    if op=='min':return np.minimum(*args)
    if op=='max':return np.maximum(*args)
    if op=='abs':return np.abs(args[0])
    if op=='sqrt':return np.sqrt(args[0])
    if op=='log':return np.log(args[0])
    if op=='neg':return -args[0]
    raise ValueError(op)

def strategy(expr,alpha):
    if expr=='lbow':return lambda_LBOW
    if expr=='predmix':return lambda x,m:lambda_predmix_eb(x,alpha=alpha)
    def fn(x,m):
        n=len(x);t=np.arange(1,n+1)
        sums=np.append(0,np.cumsum(x)[:-1]);squares=np.append(0,np.cumsum(x*x)[:-1])
        mean=(0.5+sums)/t
        features={'t':t,'n':n,'sum':sums,'sumsq':squares,'mean':mean,
                  'variance':np.maximum(0,(0.25+squares)/t-mean**2),
                  'lbow':lambda_LBOW(x,m),'predmix':lambda_predmix_eb(x,alpha=alpha)}
        with np.errstate(all='ignore'):v=np.broadcast_to(_eval(expr,features),(n,)).copy()
        if not np.all(np.isfinite(v)):raise ValueError('nonfinite predictable bet')
        return v
    return fn

def algorithm_seed(iid,bench_seed):
    # Independent of parameter identity, input row position and observation seed.
    return int.from_bytes(hashlib.sha256((str(bench_seed)+':'+str(iid)).encode()).digest()[:4],'little')

def evaluate(x,iid,bench_seed,policy,legacy_seed=None):
    n=len(x);sizes=np.linspace(100,2000,10,dtype=int)
    i=int(np.flatnonzero(sizes==n)[0])
    seed=algorithm_seed(iid,bench_seed) if legacy_seed is None else int(legacy_seed)
    x=np.array(x,dtype=np.float64,copy=True)
    theta=policy['theta']
    positive=strategy(policy['positive'],0.05*theta) if theta>0 else None
    negative=strategy(policy['negative'],0.05*(1-theta)) if theta<1 else None
    vals=[]
    for l in range(policy['B']):
        # Preserve sequential in-place shuffling and the authors' i*l seed offset.
        np.random.seed((seed+i*l)%(2**32));np.random.shuffle(x)
        vals.append(betting_mart(x,0.5,lambdas_fn_positive=positive,lambdas_fn_negative=negative,
                   theta=theta,trunc_scale=policy['trunc_scale'],convex_comb=policy['combine']=='mixture')[-1])
    e=float(np.cumsum(vals)[-1]/policy['B'])
    np.random.seed(seed);u=float(np.random.uniform(0,1,1)[-1])
    return e,u,int(e>u/0.05)

def worker(job):
    x,iid,seed,policy=job
    return str(iid),evaluate(x,iid,seed,policy)
