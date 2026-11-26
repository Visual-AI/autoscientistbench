"""Frozen output schema and declared protocol checks (no candidate imports)."""
import json
import numpy as np
# Matched experimental controls for the fixed-network dropout-allocation problem.
CONFIG=dict(depth=6,width=256,input_dim=3072,output_dim=10,epochs=75,batch_size=100,
            lr=1e-4,weight_decay=1e-7,eta_min=1e-7,sigma_w_sq=1.98,sigma_b_sq=.02,
            optimizer='AdamW',loss='CrossEntropyLoss',dtype='float32')
KEYS={'ids','logits','seeds','bench_seed','rates','config'}
OPTIONAL_KEYS=set()
def load(path):
    with np.load(path,allow_pickle=False) as z:
        if not KEYS<=set(z.files) or set(z.files)-KEYS-OPTIONAL_KEYS:raise ValueError('prediction keys mismatch')
        d={k:z[k] for k in z.files}
    ids=d['ids'];n=len(ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or n==0 or len(np.unique(ids))!=n:raise ValueError('invalid/duplicate ids')
    if not np.all(np.diff(ids)>0):raise ValueError('ids must be sorted')
    if d['logits'].shape!=(25,n,10) or d['logits'].dtype!=np.float32 or not np.isfinite(d['logits']).all():raise ValueError('logit shape/type/finiteness')
    if 'rates' in d and (d['rates'].shape!=(25,6) or d['rates'].dtype!=np.float64 or not np.isfinite(d['rates']).all()):raise ValueError('diagnostic rate shape/type/finiteness')
    if d['seeds'].shape!=(25,) or d['seeds'].dtype.kind not in 'iu':raise ValueError('seed shape/type')
    if d['bench_seed'].shape!=() or d['bench_seed'].dtype.kind not in 'iu':raise ValueError('bench_seed scalar')
    if 'config' in d:
        if d['config'].shape!=() or d['config'].dtype.kind not in 'US':raise ValueError('config scalar string')
        if not isinstance(json.loads(str(d['config'])),dict):raise ValueError('config must describe a JSON object')
    return d

def guard(d):
    seed=int(d['bench_seed']);want=np.arange(42+25*(seed-1),67+25*(seed-1))
    if seed<1 or not np.array_equal(d['seeds'],want):raise ValueError('all 25 prescribed seed replicas required')

    rates=d['rates']
    if np.any(rates<0) or np.any(rates>=1) or not np.allclose(rates.mean(axis=1),.1,rtol=0,atol=1e-12):
        raise ValueError('dropout allocation must preserve mean0.1 and rates in[0,1)')
    if not np.allclose(rates,rates[0],rtol=0,atol=1e-12):
        raise ValueError('replicas must share one dropout allocation')
    config=json.loads(str(d['config']))
    if config != CONFIG:
        raise ValueError('fixed network and matched training controls changed')
