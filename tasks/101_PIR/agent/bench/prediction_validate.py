#!/usr/bin/env python3
"""validate.py PRED INPUT; exit nonzero for invalid tensor archives."""
import sys
import numpy as np

def validate(pred_path,input_path):
    with np.load(input_path,allow_pickle=False) as inp,np.load(pred_path,allow_pickle=False) as pred:
        hs=sorted(set(inp['horizon'].tolist()))
        if set(pred.files)!={f'{k}_{h}' for h in hs for k in ('id','pred')}:raise ValueError('wrong prediction keys')
        if inp['x'].shape!=(len(inp['id']),96,7):raise ValueError('input must have 96 history steps and 7 channels')
        for h in hs:
            ids=pred[f'id_{h}'];p=pred[f'pred_{h}'];want=inp['id'][inp['horizon']==h]
            if ids.ndim!=1 or ids.dtype.kind not in 'US':raise ValueError('id must be a string vector')
            if len(set(ids.tolist()))!=len(ids):raise ValueError('duplicate ids')
            if set(ids.tolist())!=set(want.tolist()):raise ValueError('id set mismatch')
            if p.shape!=(len(ids),h,7):raise ValueError('prediction shape mismatch')
            if p.dtype.kind!='f':raise ValueError('prediction must be floating point')
            if not np.isfinite(p).all():raise ValueError('nonfinite prediction')
if __name__=='__main__':
    try:validate(sys.argv[1],sys.argv[2])
    except Exception as e:sys.exit(str(e))
    print('ok')
