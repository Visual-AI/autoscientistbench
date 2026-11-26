#!/usr/bin/env python3
"""Exact cdr_bench QNN AUC, with vectorized histogram and prefix sums.

Preserves float32 Tanimoto arithmetic, stable ranks, and removal of rank zero
(including the original behavior on duplicate fingerprints and tied coordinates).
"""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction


def ranks(d):
    return np.argsort(np.argsort(d,axis=1,kind='stable'),axis=1,kind='stable')


def high_dist(packed):
    x=np.unpackbits(packed,axis=-1,count=1024).astype(np.float32)
    intersection=x@x.T
    sums=x.sum(axis=1)
    union=sums[:,None]+sums[None,:]-intersection
    sim=np.divide(intersection,union,out=np.zeros_like(intersection),where=union!=0)
    d=(np.float32(1)-sim).astype(np.float64)
    np.fill_diagonal(d,0)
    return d


def low_dist(xy):
    mag=np.max(np.abs(xy))
    if mag>1e100 or 0<mag<1e-100: xy=xy/mag
    dx=xy[:,None,0]-xy[None,:,0]
    dy=xy[:,None,1]-xy[None,:,1]
    return dx*dx+dy*dy


def qnn_from_ranks(high,low):
    n=len(high)
    q=np.bincount((high*n+low).ravel(),minlength=n*n).reshape(n,n)
    # Integer sums are exact. Original repeated Q_inner[:k,:k].sum is cubic.
    cumulative=q[1:,1:].cumsum(0).cumsum(1)
    return cumulative.diagonal()/(np.arange(1,n,dtype=np.float64)*n)


def score(pred,labels):
    # Trusted data loading stays outside the candidate-format guard.
    with np.load(labels,allow_pickle=False) as z:
        ids,packed,offsets=[z[k] for k in ['ids','packed','offsets']]
    try:
        coords=read_prediction(pred,ids)
        if coords.shape!=(len(ids),2): raise ValueError('wrong coordinate shape')
    except (ValueError,TypeError,KeyError,OSError,OverflowError) as e:
        # Raw manifest score is zero. Infinity is the harness failure sentinel.
        return {'auc_mean':float('inf'),'score':0.0,'auc_size_weighted':0.0,'valid':False,'reason':str(e)}
    auc=[]
    for s,e in zip(offsets[:-1],offsets[1:]):
        curve=qnn_from_ranks(ranks(high_dist(packed[s:e])),ranks(low_dist(coords[s:e])))
        auc.append(float(np.mean(curve)))
    value=float(np.mean(auc))
    return {'auc_mean':value,'score':value,'auc_size_weighted':float(np.average(auc,weights=np.diff(offsets))),'valid':True,'per_cluster_auc':auc}

if __name__=='__main__': print(json.dumps(score(*sys.argv[1:])))
