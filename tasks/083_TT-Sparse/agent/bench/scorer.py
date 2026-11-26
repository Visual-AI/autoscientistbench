#!/usr/bin/env python3
"""Mean of five ROC-AUCs, guarded by extracted-rule inference and mean complexity."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from certificate import read_prediction, reconstruct, unpack

def score(pred_path,labels_path):
    ids,pred,card=read_prediction(pred_path)
    lab=pd.read_csv(labels_path,float_precision='round_trip')
    assert lab.id.is_unique and set(lab.id)==set(ids),'label ids mismatch'
    lab=lab.set_index('id').loc[ids]
    assert set(lab.y)=={0,1},'labels must contain both binary classes'
    costs=[r.complexity for r,e in unpack(card)]
    cost=float(np.mean(costs))
    if cost>14:
        return {'auc':float('inf'),'paper_score':.5,'guard_complexity_ok':False,'mean_complexity':cost,'complexities':costs}
    rec,_=reconstruct(card,lab)
    err=float(np.max(np.abs(rec-pred)))
    if not np.isfinite(rec).all() or err>1e-12:
        return {'auc':float('inf'),'paper_score':.5,'guard_rules_ok':False,'max_prediction_error':err}
    values=[float(roc_auc_score(lab.y,pred[:,s])) for s in range(5)]
    return {'auc':float(np.mean(values)),'auc_per_replicate':values,'mean_complexity':cost,'complexities':costs,'guard_complexity_ok':True,'guard_rules_ok':True,'max_prediction_error':err}

def main(pred_path,labels_path):
    try:report=score(pred_path,labels_path)
    except Exception as e:report={'auc':float('inf'),'paper_score':.5,'reason':str(e)}
    print(json.dumps(report,allow_nan=True))
if __name__=='__main__':main(*sys.argv[1:])
