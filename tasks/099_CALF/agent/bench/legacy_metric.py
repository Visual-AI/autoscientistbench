#!/usr/bin/env python3
"""Four-horizon mean MSE in training-standardized units. MAE is diagnostic only."""
import json,sys
import numpy as np
HORIZONS=(96,192,336,720)
TRAIN_MEAN=np.array([7.937742245659508, 2.0210386567335163, 5.079770601157927, 0.7461858799957015, 2.781762386375555, 0.7884531235540096, 17.1282616982271])
TRAIN_STD=np.array([5.812749409143771, 2.0901046504076, 5.518793579036245, 1.9263792741329822, 1.0235226594952194, 0.6302366362251923, 9.176491024944335])
def standardize(a):return (np.asarray(a,dtype=np.float64)-TRAIN_MEAN)/TRAIN_STD
def score(pred_path,labels_path):
    with np.load(pred_path,allow_pickle=False) as pred,np.load(labels_path,allow_pickle=False) as lab:
        mse={};mae={}
        for h in HORIZONS:
            pid=pred[f'id_{h}'];ids=lab[f'id_{h}'];p=pred[f'pred_{h}'];y=lab[f'y_{h}']
            if pid.ndim!=1 or pid.dtype.kind not in 'US':raise ValueError('ids must be a string vector')
            if len(set(pid.tolist()))!=len(pid) or set(pid.tolist())!=set(ids.tolist()):raise ValueError('id mismatch or duplicates')
            if p.shape!=y.shape or p.shape!=(len(ids),h,7):raise ValueError('incorrect shape')
            if p.dtype.kind!='f' or not np.isfinite(p).all():raise ValueError('predictions must be finite floating point')
            order={k:i for i,k in enumerate(pid.tolist())};idx=[order[k] for k in ids.tolist()]
            err=standardize(p[idx])-standardize(y)
            mse[str(h)]=float(np.mean(err**2));mae[str(h)]=float(np.mean(np.abs(err)))
        result={'mse':sum(mse.values())/4,'mae':sum(mae.values())/4,'mse_per_horizon':mse,'mae_per_horizon':mae}
        if not np.isfinite(result['mse']):raise ValueError('nonfinite metric')
        return result
def main():
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result={'mse':float('inf'),'reason':str(e)}
    print(json.dumps(result))
if __name__=='__main__':main()
