#!/usr/bin/env python3
"""Validate finite floating-point forecasts and exact, unique per-horizon ids."""
import sys
import numpy as np

def validate(pred_path,input_path):
    with np.load(input_path,allow_pickle=False) as inp,np.load(pred_path,allow_pickle=False) as pred:
        for h in sorted(set(inp['horizon'].tolist())):
            ids=pred[f'id_{h}'];p=pred[f'pred_{h}'];want=inp['id'][inp['horizon']==h]
            if ids.ndim!=1 or ids.dtype.kind not in 'US':raise ValueError('ids must be a string vector')
            got=ids.tolist()
            if len(got)!=len(set(got)):raise ValueError('duplicate ids')
            if set(got)!=set(want.tolist()):raise ValueError('id set mismatch')
            if p.shape!=(len(ids),h,7):raise ValueError('incorrect prediction shape')
            if p.dtype.kind!='f' or not np.isfinite(p).all():raise ValueError('predictions must be finite floating point')
    print('ok')
if __name__=='__main__':
    try:validate(sys.argv[1],sys.argv[2])
    except Exception as e:sys.exit(str(e))
