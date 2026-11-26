#!/usr/bin/env python3
"""Reconstruct the fixed model, compute target-averaged MSE, then average runs0–4."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from format import LABEL, PARAMS, read, aligned
from hit_rate_model import HitRateModel

def score(pred_path,labels_path):
    labels=read(labels_path,LABEL)
    predictions=aligned(pred_path,labels)
    groups={}
    for l,p in zip(labels,predictions):
        groups.setdefault((int(l['run']),l['target']),[]).append((l,p))
    expected={(s,t) for s in range(5) for t in ('ampc','d4','sigma2')}
    if set(groups)!=expected: raise ValueError('missing run or target')
    values={}
    for (s,t),rows in sorted(groups.items()):
        y=np.array([float(l['hit_rate_mean']) for l,_ in rows])
        params={key:float(rows[0][1][key]) for key in PARAMS}
        pprop=np.array([float(l['pprop']) for l,_ in rows])
        definition=np.array([float(l['definition']) for l,_ in rows])
        yhat=HitRateModel(params).get_pprop_hr(pprop,definition)
        claimed=np.array([float(p['pred']) for _,p in rows])
        if not np.isfinite(yhat).all() or not np.allclose(claimed,yhat,rtol=1e-10,atol=1e-12):
            raise ValueError('predictions do not match fixed six-parameter model')
        values[f'{s}/{t}']=float(np.mean((yhat-y)**2))
    per_run=[float(np.mean([values[f'{s}/{t}'] for t in ('ampc','d4','sigma2')])) for s in range(5)]
    return dict(mse=float(np.mean(per_run)),per_run=per_run,per_target={t:float(np.mean([values[f'{s}/{t}'] for s in range(5)])) for t in ('ampc','d4','sigma2')},guards='pass')
if __name__=='__main__':
    try: result=score(*sys.argv[1:])
    except Exception as e: result={'mse':float('inf'),'guard_failure':str(e)}
    print(json.dumps(result))
