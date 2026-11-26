"""Frozen WS graph generator, identical RNG order to the reproduction.
Only the offline data builder may simulate opinions. Predictors may not use
simulation, MC-fitted tables, or validation truth to fit their output curves.
The public fixed topology parameters define a distribution, not a label table.
"""
import numpy as np
n=320
k=14
BETA=.115

def ws(beta,seed):
    rng=np.random.default_rng(seed); a=np.zeros((n,n),dtype=bool)
    for d in range(1,k//2+1):
        for u in range(n): a[u,(u+d)%n]=a[(u+d)%n,u]=True
    # NetworkX WS edge order: distance first, then node; uniform rejection draw.
    for d in range(1,k//2+1):
        for u in range(n):
            v=(u+d)%n
            if rng.random()<beta:
                if a[u].sum()>=n-1: continue
                w=int(rng.integers(n))
                while w==u or a[u,w]: w=int(rng.integers(n))
                a[u,v]=a[v,u]=False; a[u,w]=a[w,u]=True
    return a

def stats(a):
    deg=a.sum(axis=1); cc=[]
    for u in range(n):
        ids=np.flatnonzero(a[u]); cc.append(float(a[np.ix_(ids,ids)].sum()/(deg[u]*(deg[u]-1))))
    return float(deg.mean()),float(np.mean(cc)),int(deg.min())

def adjacency_text(seed):
    a=ws(BETA,int(seed))
    return '\n'.join(' '.join(map(str,np.flatnonzero(row))) for row in a)+'\n'
