"""Equal-weight normalized Winkler with the mandatory per-cell coverage floor."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from numeric_validate import read_prediction

COVERAGE_FLOOR={'test':.86,'val':.83}

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

def score(pred_path,labels_path):
    ids,p=read_prediction(pred_path)
    a=np.load(labels_path,allow_pickle=False)
    expected=a['id'].tolist()
    if set(ids)!=set(expected) or len(ids)!=len(expected):raise ValueError('id set mismatch')
    lookup={x:i for i,x in enumerate(ids)}
    with np.errstate(over='ignore', invalid='ignore'):
        p=p[[lookup[x] for x in expected]].astype(np.float32)
    if not np.isfinite(p).all():raise ValueError('intervals exceed float32 range')
    y=a['y'].astype(np.float32);mask=a['mask'].astype(bool)
    low,high=p.T
    width=high-low
    winkler=width+np.float32(20)*((low-y)*(y<low)+(y-high)*(y>high))
    covered=(low<=y)&(y<=high)
    metrics={};ratios=[]
    for cell,denom in zip(a['cells'].tolist(),a['denominator']):
        keep=(a['cell']==cell)&mask
        if not np.any(keep) or not np.isfinite(y[keep]).all() or not denom>0:raise ValueError('invalid labels')
        cov=float(covered[keep].astype(np.float32).mean())
        w=float(winkler[keep].mean())
        if not np.isfinite(w):raise ValueError('Winkler arithmetic overflow')
        metrics[cell]={'coverage':cov,'winkler':w,'width':float(width[keep].mean()),'ratio':w/float(denom)}
        ratios.append(w/float(denom))
    if len(metrics)!=12:raise ValueError('expected 12 cells')
    floor=COVERAGE_FLOOR[split_of(labels_path)]
    ok=all(v['coverage']>=floor for v in metrics.values())
    value=float(np.mean(ratios))
    if not np.isfinite(value):raise ValueError('normalized Winkler arithmetic overflow')
    return {'normalized_winkler':value if ok else float('inf'),'unguarded':value,'coverage_ok':ok,'coverage_floor':floor,'cells':metrics}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'normalized_winkler':float('inf'),'reason':str(e)}
    print(json.dumps(result,allow_nan=True))
