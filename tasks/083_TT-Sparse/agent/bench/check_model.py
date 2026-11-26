#!/usr/bin/env python3
"""Judge probe: prediction certificate must equal the train-time rules.json artifact."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from certificate import read_prediction,reconstruct

def check(model,input_,pred):
    card=json.loads((Path(model)/'rules.json').read_text())
    ids,p,pcard=read_prediction(pred)
    assert card==pcard,'prediction rules differ from train-time certificate'
    frame=pd.read_csv(input_,float_precision='round_trip').set_index('id')
    assert frame.index.is_unique and set(frame.index)==set(ids),'input ids mismatch'
    recon,cost=reconstruct(card,frame.loc[ids]);err=float(np.max(np.abs(recon-p)))
    assert np.isfinite(recon).all() and err<=1e-12,'not frozen predict_rules output'
    assert np.mean(cost)<=14,'mean complexity exceeds 14'
    return {'pass':True,'max_prediction_error':err,'complexities':cost}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--input',required=True);p.add_argument('--pred',required=True);p.add_argument('--report');a=p.parse_args()
    try:r=check(a.model,a.input,a.pred)
    except Exception as e:r={'pass':False,'auc':float('inf'),'reason':str(e)}
    if a.report:Path(a.report).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r));sys.exit(0 if r['pass'] else 1)
