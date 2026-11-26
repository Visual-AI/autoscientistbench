"""Strict NPZ tensor contract. Usage: validate.py PRED INPUT."""
import sys
import numpy as np
KEYS={'id','w','v','theta','oracle_counts'}

def read_prediction(path, expected_ids):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=KEYS:raise ValueError('prediction keys must be id,w,v,theta,oracle_counts')
        p={k:z[k] for k in KEYS}
    n=len(expected_ids)
    if p['id'].dtype.kind not in 'US' or p['id'].shape!=(n,):raise ValueError('invalid id array')
    ids=p['id'].tolist()
    if len(set(ids))!=n:raise ValueError('duplicate id')
    if ids!=list(expected_ids):raise ValueError('id set/order mismatch')
    for k,shape,kinds in [('w',(n,10,40,784),'f'),('v',(n,10,2,40),'f'),('theta',(n,2),'f'),('oracle_counts',(n,500,2),'fi')]:
        a=p[k]
        if a.shape!=shape or a.dtype.kind not in kinds:raise ValueError(f'invalid {k} shape/dtype')
        if not np.isfinite(a).all():raise ValueError(f'nonfinite {k}')
    c=p['oracle_counts']
    if (c<0).any() or (c!=np.floor(c)).any():raise ValueError('counts must be nonnegative integers')
    return p

def validate(pred_path,input_path):
    with np.load(input_path,allow_pickle=False) as z:
        return read_prediction(pred_path,z['id'])
if __name__=='__main__':
    try:
        p=validate(sys.argv[1],sys.argv[2]); print(f"ok: {len(p['id'])} trials")
    except Exception as e:sys.exit(str(e))
