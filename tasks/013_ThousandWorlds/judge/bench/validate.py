"""Strict tensor format validator. Usage: validate.py PRED INPUT."""
import sys
import numpy as np
SUBSETS = ('multi-partial', 'multi-complete', 'single-complete')

def check(pred, inputs):
    expected={s+'__'+k for s in SUBSETS for k in ['id','pred']}
    if set(pred.files)!=expected: raise ValueError('incorrect prediction keys')
    for s in SUBSETS:
        ids=pred[s+'__id']; ref=inputs[s+'__id']; p=pred[s+'__pred']
        if ids.ndim!=1 or ids.dtype.kind not in 'iu':raise ValueError('ids must be integer vector')
        if len(np.unique(ids))!=len(ids):raise ValueError('duplicate ids')
        if not np.array_equal(ids,ref):raise ValueError('ids/order mismatch')
        f=len(inputs[s+'__field_names'])
        if p.shape!=(5,len(ids),f,32,64):raise ValueError('incorrect prediction shape')
        if p.dtype.kind!='f' or not np.isfinite(p).all():raise ValueError('predictions must be finite floats')
    return True

def main(pred_path,input_path):
    with np.load(pred_path,allow_pickle=False) as p,np.load(input_path,allow_pickle=False) as x:check(p,x)
    print('ok')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
