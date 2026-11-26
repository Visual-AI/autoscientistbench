"""Fold mean MAE, gated by baseline-relative HGR/GDP/PF; fairness fallback 0.1785."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from validate import read_predictions
from metrics import evaluate

def score(pred,labels):
    ids,values=read_predictions(pred)
    with open(labels,newline='') as f:rows=list(csv.DictReader(f))
    if ids!=[r['id'] for r in rows]:raise ValueError('id set or order mismatch')
    groups=list(dict.fromkeys(r['fold'] for r in rows))
    arr=[]
    for group in groups:
        idx=[i for i,r in enumerate(rows) if r['fold']==group]
        arr.append(evaluate([values[i] for i in idx],[float(rows[i]['y']) for i in idx],[float(rows[i]['p']) for i in idx]))
    mean=np.mean(arr,axis=0)
    caps=np.asarray([[float(r[k]) for k in ('hgr_cap','gdp_cap','pf_cap')] for r in rows])
    if not np.isfinite(mean).all() or not np.isfinite(caps).all():raise ValueError('nonfinite metric or cap')
    if not np.all(caps==caps[0]):raise ValueError('inconsistent caps')
    passed=bool(np.all(mean[1:]<=caps[0]))
    return {'fair_mae':float(mean[0]) if passed else 0.1785,'mae':float(mean[0]),'hgr':float(mean[1]),'gdp':float(mean[2]),'pf':float(mean[3]),'fairness_pass':passed,'fold_metrics':np.asarray(arr).tolist()}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'fair_mae':float('inf'),'invalid':str(e)}
    print(json.dumps(result))
