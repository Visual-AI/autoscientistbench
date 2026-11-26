"""Validate full finite sample tensors, exact instance IDs, shapes and NPZ keys."""
import sys, re
import numpy as np

def load_prediction(path, shapes):
    with np.load(path,allow_pickle=False) as z:
        assert len(z.files)==len(set(z.files)), 'duplicate ZIP keys'
        ids=z['ids'];assert ids.ndim==1 and ids.dtype.kind in 'US','ids must be a string vector'
        ids=ids.tolist();assert len(ids)==len(set(ids)),'duplicate ids'
        assert set(ids)==set(shapes),'instance ids differ'
        assert set(z.files)=={'ids',*ids},'unexpected or missing sample arrays'
        result={}
        for iid in ids:
            x=z[iid]
            assert x.dtype.kind=='f' and x.shape==shapes[iid],f'incorrect dtype/shape for {iid}'
            assert np.isfinite(x).all(),f'nonfinite sample for {iid}'
            result[iid]=x
        return result

def main(pred,input_path):
    with np.load(input_path,allow_pickle=False) as z:
        shapes={}
        for iid in z['ids'].tolist():
            D=z[iid+'__dataset'].shape[1]
            shapes[iid]=({2:25000,4:25000,6:30000,10:40000,16:40000}[D],D)
    result=load_prediction(pred,shapes)
    print(f'ok: {len(result)} complete sample tensors')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:print(f'invalid: {e}',file=sys.stderr);sys.exit(1)
