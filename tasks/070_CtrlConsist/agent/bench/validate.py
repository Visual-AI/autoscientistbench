"""Strict tensor prediction schema; row order is arbitrary, IDs are opaque."""
import sys
import numpy as np

def read_prediction(path, ids):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files) != {'id','drift'}: raise ValueError('keys must be id,drift')
        pid,d=z['id'],z['drift']
    if pid.ndim != 1 or pid.dtype.kind not in 'US': raise ValueError('id must be a string vector')
    if len(set(pid.tolist())) != len(pid): raise ValueError('duplicate IDs')
    if len(pid)!=len(ids) or set(pid.tolist())!=set(ids.tolist()): raise ValueError('ID set mismatch')
    if d.shape != (len(ids),500,1) or d.dtype.kind != 'f': raise ValueError('drift shape/dtype mismatch')
    if not np.isfinite(d).all(): raise ValueError('nonfinite drift')
    lookup={s:i for i,s in enumerate(pid.tolist())}
    return d[[lookup[s] for s in ids.tolist()]]

def main(pred,inputs):
    with np.load(inputs,allow_pickle=False) as z:ids=z['id']
    read_prediction(pred,ids)
    print('ok')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
