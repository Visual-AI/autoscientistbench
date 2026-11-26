"""Validate a tensor of joint posterior draws, keyed by opaque instance id."""
import sys
import numpy as np


def read_pred(path, ids):
    with np.load(path,allow_pickle=False) as p:
        if set(p.files)!={'id','z','seconds'}:raise ValueError('keys must be id,z,seconds')
        ident,z,t=p['id'],p['z'],p['seconds']
    if ident.ndim!=1 or ident.dtype.kind not in 'US':raise ValueError('id must be a string vector')
    if len(set(ident))!=len(ident) or set(ident)!=set(ids):raise ValueError('id mismatch or duplicate')
    if z.shape!=(len(ids),8000,10) or t.shape!=(len(ids),):raise ValueError('wrong posterior/time shape')
    if z.dtype.kind!='f' or t.dtype.kind!='f':raise ValueError('numeric arrays must be floating point')
    if not np.isfinite(z).all() or not np.isfinite(t).all() or np.any(t<=0):raise ValueError('non-finite samples or non-positive elapsed time')
    return {str(i):(z[j],float(t[j])) for j,i in enumerate(ident)}


def main(pred,input_path):
    with np.load(input_path,allow_pickle=False) as f:ids=f['id']
    read_pred(pred,ids)
    print('ok: posterior shapes, ids, finite samples and positive times')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
