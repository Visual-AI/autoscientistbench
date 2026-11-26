#!/usr/bin/env python3
"""Mean of ten run accuracies, with RDP accounting. Invalid => failure sentinel."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from common import read_predictions,privacy

def score(pred_path,labels_path):
    ids,p,h,d=read_predictions(pred_path)
    with np.load(labels_path,allow_pickle=False) as a:
        target_ids=a['id'];y=a['y']
        if target_ids.ndim!=1 or len(set(target_ids.tolist()))!=len(target_ids) or not np.array_equal(ids,target_ids):raise ValueError('label id mismatch')
        # A matrix is accepted for the operator archive audit of independent splits.
        if y.ndim==1:y=y[:,None]
        if y.shape not in [(len(ids),1),(len(ids),10)] or not np.isfinite(y).all() or not np.equal(y,y.astype(int)).all() or not ((0<=y)&(y<26)).all():raise ValueError('invalid labels')
    eps=privacy(h,d)
    values=np.mean(p==y,axis=0)
    return {'mean_accuracy':float(np.mean(values)),'per_run_accuracy':values.tolist(),'epsilon_max':max(eps),'valid':True}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'mean_accuracy':float('inf'),'rule_score':1/26,'valid':False,'reason':str(e)}
    print(json.dumps(result))
