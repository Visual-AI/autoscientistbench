"""Original full-matrix AUC, including diagonal negatives and original tie order."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from validate import read_prediction
from roc import compute_roc_curve,auc_trapezoid

def score(pred_path,labels_path):
    with np.load(labels_path,allow_pickle=False) as z:
        order=np.argsort(z['trial'])
        ids=z['id'][order]; truth=z['true_prec'][order]
    pred=read_prediction(pred_path,ids)
    if truth.shape!=pred.shape or not np.isfinite(truth).all():raise ValueError('invalid labels')
    values=[]
    for p,t in zip(pred,truth):
        fpr,tpr=compute_roc_curve(np.abs(t)>1e-5,np.abs(p))
        values.append(auc_trapezoid(fpr,tpr))
    return dict(edge_auc=float(np.mean(values)),instance_std=float(np.std(values)),per_instance=values)

if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result=dict(edge_auc=float('inf'),error=str(e))
    print(json.dumps(result))
