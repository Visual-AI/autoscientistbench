#!/usr/bin/env python3
"""Verify the fixed affine research object and its declared privacy ledger."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from common import RUNS,privacy,read_predictions

def check(model,input_path,pred_path):
    c=json.loads((Path(model)/'model_card.json').read_text())
    if c['format'] != 'dp-affine/v1' or c['n_features']!=617 or c['n_classes']!=26 or len(c['runs'])!=RUNS:raise ValueError('incorrect classifier schema')
    ids,p,h,d=read_predictions(pred_path)
    with np.load(input_path,allow_pickle=False) as a:
        if set(a.files)!={'id','X'}:raise ValueError('prediction input must contain only id and X')
        if not np.array_equal(ids,a['id']):raise ValueError('id mismatch')
        X=a['X']
        if X.shape!=(len(ids),617) or not np.isfinite(X).all():raise ValueError('invalid observed features')
    for i,r in enumerate(c['runs']):
        if r['run']!=i:raise ValueError('run identity mismatch')
        if r['history']!=h[i] or r['delta']!=d[i]:raise ValueError('certificate/CSV ledger mismatch')
    eps=privacy(h,d)
    # The raw-feature affine family is the research object, independently of
    # which valid private solver produced its weights.
    import torch
    torch.set_num_threads(1)
    X=torch.tensor(X,dtype=torch.float32)
    for i,r in enumerate(c['runs']):
        W=np.asarray(r['weight']);b=np.asarray(r['bias'])
        if W.shape!=(26,617) or b.shape!=(26,) or not np.isfinite(W).all() or not np.isfinite(b).all():raise ValueError('predictor must be finite affine 617->26')
        expected=torch.nn.functional.linear(X,torch.tensor(W,dtype=torch.float32),torch.tensor(b,dtype=torch.float32)).argmax(1).numpy()
        if not np.array_equal(expected,p[:,i]):raise ValueError('predictions do not match affine certificate')
    return {'valid':True,'runs':RUNS,'epsilon_max':max(eps),'model_format':c['format']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--input',required=True);p.add_argument('--pred',required=True);a=p.parse_args()
    try:result=check(a.model,a.input,a.pred);rc=0
    except Exception as e:result={'valid':False,'mean_accuracy':float('inf'),'rule_score':1/26,'reason':str(e)};rc=1
    print(json.dumps(result));sys.exit(rc)
