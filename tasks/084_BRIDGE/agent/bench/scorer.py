#!/usr/bin/env python3
"""Accuracy (%) if linear weighted kappa >= .22 and macro F1 >= .27, else 0."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from table_io import read_table
import numpy as np
from sklearn.metrics import cohen_kappa_score,f1_score


def score(pred_path,labels_path):
    ids,p=read_table(pred_path,['id','pred'],'pred')
    true_ids,y=read_table(labels_path,['id','y'],'y')
    if ids!=true_ids:raise ValueError('prediction/label id mismatch')
    # pd.cut([0,15,60,240,inf], include_lowest=True): right-closed boundaries.
    p=np.searchsorted([15.,60.,240.],p,side='left')
    y=np.searchsorted([15.,60.,240.],y,side='left')
    accuracy=float(100*np.mean(y==p))
    f1=float(f1_score(y,p,labels=[0,1,2,3],average='macro',zero_division=0))
    kappa=float(cohen_kappa_score(y,p,weights='linear'))
    passed=math.isfinite(kappa) and kappa>=.22 and f1>=.27
    return {'bucket_accuracy':accuracy if passed else 0.0,'accuracy_percent':accuracy,'macro_f1':f1,'weighted_kappa':kappa if math.isfinite(kappa) else None,'quality_guards_pass':passed}

if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result={'bucket_accuracy':float('inf'),'invalid':str(e)}
    print(json.dumps(result))
