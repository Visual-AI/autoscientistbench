"""validate.py PRED INPUT; tensors are float32 and IDs preserve input order."""
import sys
import numpy as np

def validate(pred,inp):
    with np.load(inp,allow_pickle=False) as z:ids=z['id'];xshape=z['x'].shape
    with np.load(pred,allow_pickle=False) as z:
        if len(z.files)!=4 or set(z.files)!={'id','centers','parameters','inference_s'}:raise ValueError('incorrect keys')
        pid=z['id'];c=z['centers'];p=z['parameters'];t=z['inference_s']
    if pid.ndim!=1 or len(set(pid.tolist()))!=len(pid):raise ValueError('duplicate or invalid IDs')
    if not np.array_equal(pid,ids):raise ValueError('ID order/set mismatch')
    if c.shape!=(3,len(ids),21,10,32) or c.dtype!=np.float32:raise ValueError('centers must be float32 [3,N,21,10,32]')
    if p.shape!=(3,) or p.dtype.kind not in 'iu' or (p<0).any():raise ValueError('invalid parameter counts')
    if t.shape!=(3,) or (t<0).any():raise ValueError('invalid inference durations')
    if not all(np.isfinite(v).all() for v in (c,p,t)):raise ValueError('nonfinite output')
    return pid,c,p,t
if __name__=='__main__':
    try:validate(*sys.argv[1:]);print('ok')
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
