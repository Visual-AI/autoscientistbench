#!/usr/bin/env python3
"""Mean of ten float32 MAEs, under the fixed 128-real-parameter capacity budget."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import predictions,read_csv

def score(pred_path,labels_path):
    ids,p,counts=predictions(pred_path)
    rows=read_csv(labels_path,['id','y'])
    if ids!=[r['id'] for r in rows]:raise ValueError('labels id/order mismatch')
    y=np.asarray([float(r['y']) for r in rows],dtype=np.float32)
    if not np.isfinite(y).all():raise ValueError('non-finite labels')
    # Each contiguous column is reduced independently, matching the released
    # evaluator's float32 abs(...).mean(); outer aggregation is float64.
    maes=[float(np.abs(p[:,s].copy()-y).mean()) for s in range(10)]
    valid=bool((counts<=128).all())
    return {'mae':float(np.mean(maes)) if valid else float('inf'),
            'per_model_mae':maes,'real_parameters':counts.astype(int).tolist(),
            'reported_parameter_guard':valid}
if __name__=='__main__':
    try:r=score(*sys.argv[1:])
    except Exception as e:r={'mae':float('inf'),'reason':str(e)}
    print(json.dumps(r))
