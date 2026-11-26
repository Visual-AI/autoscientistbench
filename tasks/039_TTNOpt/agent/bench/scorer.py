#!/usr/bin/env python3
"""Sec. 4.2 dense squared fidelity; structural violations fail the submission."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from network import read_prediction,dense_fidelity

def score(pred,labels):
    with np.load(labels,allow_pickle=False) as data:
        ids=data['ids'].tolist();records=read_prediction(pred,ids)
        values=[];bonds=[]
        for i,name in enumerate(ids):
            f,b=dense_fidelity(records[name],data[f'target_{i}'])
            values.append(f);bonds.append(b)
    return {'fidelity':float(np.mean(values)),'per_instance':values,'max_bond':max(bonds),'is_binary_tree':True}

if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result={'fidelity':float('inf'),'guard_violation':str(e),'worst_score':0.0}
    print(json.dumps(result))
