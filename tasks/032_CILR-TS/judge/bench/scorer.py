"""MCC over independent instances. Invalid matrices have the binding worst score 0.

Implements absolute Pearson correlation with exact optimal assignment.
The authors' metric is also reported for archive comparison, without changing
any model or training code. No model code is imported.
Observation-only provenance is a patch-audit rule, not inferable from arrays.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction
from legacy_metric import mean_corr_coef


def score(pred_path,truth_path):
    with np.load(truth_path,allow_pickle=False) as z:
        ids=z['id'];truth=z['factors']
    if truth.shape!=(len(ids),200,5) or not np.isfinite(truth).all():
        raise ValueError('invalid frozen labels')
    try:
        pred=read_prediction(pred_path,ids)
    except Exception as e:
        return dict(mcc=0.0,guards_ok=False,reason=str(e))
    vals=[];legacy=[]
    for f,p in zip(truth,pred):
        # Pearson of a constant column is assigned zero association. This makes
        # collapsed factors attain the binding worst score without NaN.
        # Rescale before centering to avoid overflow for finite large outputs.
        f=f/np.maximum(np.max(np.abs(f),axis=0),1.0)
        original_p=p
        p=p/np.maximum(np.max(np.abs(p),axis=0),1.0)
        fc=f-f.mean(axis=0);pc=p-p.mean(axis=0)
        fn=np.linalg.norm(fc,axis=0);pn=np.linalg.norm(pc,axis=0)
        den=fn[:,None]*pn[None,:]
        corr=np.divide(fc.T@pc,den,out=np.zeros((5,5)),where=den>0)
        corr=np.clip(np.abs(corr),0,1)
        rows,cols=linear_sum_assignment(-corr)
        vals.append(float(corr[rows,cols].mean()))
        if np.max(np.abs(original_p)) <= np.finfo(np.float32).max:
            legacy.append(float(mean_corr_coef(torch.from_numpy(truth[len(vals)-1]).float(),
                                               torch.from_numpy(original_p).float()).item()))
        else:
            legacy.append(float('nan'))
    return dict(mcc=float(np.mean(vals)),per_instance=vals,instance_std=float(np.std(vals)),
                authors_mcc=float(np.mean(legacy)) if np.isfinite(legacy).all() else None,guards_ok=True)

if __name__=='__main__':
    print(json.dumps(score(sys.argv[1],sys.argv[2])))
