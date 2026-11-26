#!/usr/bin/env python3
"""Four-horizon standardized MSE. Behavioral constraints also require probes/audit."""
import json,sys
import numpy as np
HS=(96,192,336,720)
TRAIN_MEAN=np.array([7.937742245549186,2.0210386567124052,5.079770601170338,0.7461858799814872,2.7817623863756924,0.788453123560763,17.128261698278603])
TRAIN_STD=np.array([5.812749409078979,2.0901046503913965,5.518793579069257,1.9263792741210177,1.023522659511737,0.6302366362134583,9.176491025127563])
def standardize(x):return ((np.asarray(x,dtype=np.float64)-TRAIN_MEAN)/TRAIN_STD).astype(np.float32)

def score(pred_path,labels_path):
    with np.load(pred_path,allow_pickle=False) as pred,np.load(labels_path,allow_pickle=False) as lab:
        mse={};mae={}
        expected={f'{kind}_{h}' for h in HS for kind in ('id','pred')}
        if set(pred.files)!=expected:raise ValueError('wrong prediction keys')
        for h in HS:
            ids=pred[f'id_{h}'];want=lab[f'id_{h}'];p=pred[f'pred_{h}'];y=lab[f'y_{h}']
            if ids.ndim!=1 or ids.dtype.kind not in 'US' or len(set(ids.tolist()))!=len(ids):raise ValueError('duplicate or malformed ids')
            if set(ids.tolist())!=set(want.tolist()):raise ValueError('id set mismatch')
            if p.shape!=(len(ids),h,7) or p.dtype.kind!='f' or not np.isfinite(p).all():raise ValueError('shape/dtype/nonfinite violation')
            order={v:i for i,v in enumerate(ids.tolist())};p=p[[order[v] for v in want.tolist()]]
            err=standardize(p)-standardize(y)
            mse[str(h)]=float(np.mean(err**2));mae[str(h)]=float(np.mean(np.abs(err)))
        result={'mse':float(np.mean(list(mse.values()))),'mae':float(np.mean(list(mae.values()))),'mse_per_horizon':mse,'mae_per_horizon':mae,'format_guard_ok':True}
        if not np.isfinite(result['mse']):raise ValueError('nonfinite metric')
        return result
if __name__=='__main__':
    try:r=score(sys.argv[1],sys.argv[2])
    except Exception as e:r={'mse':float('inf'),'format_guard_ok':False,'reason':str(e)}
    print(json.dumps(r))
