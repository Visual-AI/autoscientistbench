#!/usr/bin/env python3
"""NPZ prediction schema check, independent of candidate code."""
import sys
import numpy as np

def read_prediction(path):
    with np.load(path,allow_pickle=False) as p:
        assert set(p.files)=={'ids','pred','sensor_locs','lags'},'prediction keys'
        p={k:p[k] for k in p.files}
    ids=p['ids'];pred=p['pred'];loc=p['sensor_locs'];lags=p['lags']
    assert ids.ndim==1 and ids.dtype.kind in 'US' and len(ids)>0,'ids must be nonempty strings'
    assert len(set(ids.tolist()))==len(ids),'duplicate ids'
    assert pred.shape==(3,len(ids),64,64),'pred shape must be (3,N,64,64)'
    assert pred.dtype.kind=='f' and np.isfinite(pred).all(),'predictions must be finite floats'
    assert loc.ndim==2 and loc.shape[1]==2 and loc.dtype.kind in 'iu','sensor locations must be integer pairs'
    assert np.all((loc>=0)&(loc<64)) and len(np.unique(loc,axis=0))==len(loc),'invalid/duplicate sensor locations'
    assert lags.shape==() and lags.dtype.kind in 'iu' and int(lags)>=0,'lags must be nonnegative integer scalar'
    return p

def validate(pred_path,input_path):
    p=read_prediction(pred_path)
    with np.load(input_path,allow_pickle=False) as x:
        ids=x['ids']
        assert len(set(ids.tolist()))==len(ids),'duplicate input ids'
        assert np.array_equal(p['ids'],ids),'ids must match input order exactly'
    return p

if __name__=='__main__':
    try:
        p=validate(sys.argv[1],sys.argv[2]);print('ok')
    except Exception as e:
        print(f'invalid: {type(e).__name__}: {e}',file=sys.stderr);sys.exit(1)
