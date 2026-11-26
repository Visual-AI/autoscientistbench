"""Mean CI length, iff coverage reaches the floor of the split."""
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction


COVERAGE_FLOOR={'test':0.80,'val':0.75}


def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'


def score(pred_path,labels_path):
    try:
        pred=read_prediction(pred_path)
        with np.load(labels_path,allow_pickle=False) as z:
            ids=[str(x) for x in z['id']]; truth=z['truth']
        if set(ids)!=set(pred) or len(ids)!=len(set(ids)):
            raise ValueError('id set mismatch')
        widths=[]; covers=[]
        for iid,target in zip(ids,truth):
            _,lo,hi=pred[iid]
            widths.append(hi-lo); covers.append(lo<=float(target)<=hi)
        raw=float(np.mean(widths)); coverage=float(np.mean(covers))
        guards=coverage>=COVERAGE_FLOOR[split_of(labels_path)]
        return {'ci_length_gmean':raw if guards else float('inf'),
                'raw_ci_length_gmean':raw,'coverage':coverage,'guards_ok':bool(guards),'n':len(ids)}
    except Exception as e:
        return {'ci_length_gmean':float('inf'),'guards_ok':False,'error':str(e)}


if __name__=='__main__':
    print(json.dumps(score(*sys.argv[1:])))
