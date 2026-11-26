#!/usr/bin/env python3
"""Five-run Hungarian ACC, gated by mean macro-F1 and exactly five clusters."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score,f1_score
from format import read_predictions

def matched_metrics(y,p):
    # Fixed class universe permits scoring a mask missing a class, without
    # inserting artificial assignments. On all-five-class inputs this is the
    # released Munkres matching and sklearn ACC / macro-F1 calculation.
    cost=np.zeros((5,5),dtype=int)
    np.add.at(cost,(y,p),1)
    match=Munkres().compute((-cost).tolist())
    mapping={j:i for i,j in match}
    aligned=np.array([mapping[int(v)] for v in p])
    return float(accuracy_score(y,aligned)),float(f1_score(y,aligned,average='macro',zero_division=0))

def score(pred_path,labels_path):
    pred=read_predictions(pred_path)
    with open(labels_path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','y']:raise ValueError('invalid labels header')
        rows=list(r)
    ids=[r['id'] for r in rows]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(pred):raise ValueError('label ids invalid')
    y=np.array([int(r['y']) for r in rows])
    if not np.isin(y,np.arange(5)).all():raise ValueError('invalid labels')
    full=np.array(list(pred.values()))
    counts=[len(np.unique(full[:,i])) for i in range(5)]
    acc=[];f1=[]
    for i in range(5):
        a,b=matched_metrics(y,np.array([pred[j][i] for j in ids]));acc.append(a);f1.append(b)
    raw=float(np.mean(acc));mean_f1=float(np.mean(f1))
    valid=counts==[5]*5 and mean_f1>=0.30
    return {'acc':raw if valid else float('inf'),'raw_acc':raw,'macro_f1':mean_f1,'acc_per_run':acc,'f1_per_run':f1,'cluster_counts':counts,'guard_pass':valid}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'acc':float('inf'),'guard_pass':False,'error':str(e)}
    print(json.dumps(result))
