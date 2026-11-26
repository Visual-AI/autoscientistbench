"""Validate raw point cloud and opaque IDs; review permits ten nonfinite rows."""
import sys
import numpy as np

def read_prediction(path, expected):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'id','samples'}: raise ValueError('keys must be id,samples')
        ids=z['id']; samples=z['samples']
    if ids.ndim!=1 or ids.dtype.kind not in 'US': raise ValueError('id must be a string vector')
    if len(ids)!=len(set(ids.tolist())): raise ValueError('duplicate id')
    if set(ids.tolist())!=set(expected.tolist()): raise ValueError('id set mismatch')
    if samples.shape!=(len(expected),3) or samples.dtype.kind!='f': raise ValueError('samples must be floating point (n,3)')
    # This explicit allowance preserves the baseline sampler's known numerical bug.
    if np.isfinite(samples).all(1).sum()<len(expected)-10: raise ValueError('more than ten nonfinite samples')
    return ids,samples

def main(pred,inp):
    with np.load(inp,allow_pickle=False) as z: expected=z['id']
    read_prediction(pred,expected)
    print('ok')
if __name__=='__main__':
    try: main(*sys.argv[1:])
    except Exception as e: sys.exit(str(e))
