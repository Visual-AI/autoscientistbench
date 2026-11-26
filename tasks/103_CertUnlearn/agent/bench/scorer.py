#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from prediction import read,accuracy_curve,crossing
from certificate import verify

def score(pred_path,label_path):
    try:
        with np.load(label_path,allow_pickle=False) as labels:
            p,c=read(pred_path,labels['id']);curve=accuracy_curve(p,labels['y'],int(labels['batch_size']))
        privacy=verify(c)
        return dict(epochs_to_50pct=crossing(curve),guard_ok=True,accuracy_curve=curve.tolist(),integer_crossing=crossing(curve,False),**privacy)
    except Exception as e:
        return dict(epochs_to_50pct=float('inf'),guard_ok=False,reason=str(e))
if __name__=='__main__':print(json.dumps(score(*sys.argv[1:3])))
