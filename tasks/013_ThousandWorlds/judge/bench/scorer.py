"""Table 4 GPLFR score: five-seed mean, literal fixed guard thresholds."""
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from metrics import _metric_space, rmse, VARIABLE_ORDER
from validate import SUBSETS, check

def score(pred_path,labels_path):
    result={}; diagnostics={}
    with np.load(pred_path,allow_pickle=False) as p,np.load(labels_path,allow_pickle=False) as y:
        check(p,y)
        for s in SUBSETS:
            names=y[s+'__field_names'].tolist();truth=_metric_space(y[s+'__Y'],names,humidity='dex')
            mask=y[s+'__mask'];norm=y[s+'__normalizer']
            if norm.shape!=(8,) or not np.isfinite(norm).all() or np.any(norm<=0):raise ValueError('invalid normalizer')
            scores=[];variables=[]
            for grid in p[s+'__pred']:
                m=rmse(_metric_space(grid,names,humidity='dex'),truth,mask,names,dex_specific_humidity=True)['per_variable']
                vals=[m[k] for k in VARIABLE_ORDER]
                if not all(math.isfinite(x) and x>0 for x in vals):raise ValueError('invalid RMSE')
                scores.append(math.exp(statistics.mean(math.log(v/b) for v,b in zip(vals,norm))))
                variables.append(m)
            result[s]=statistics.mean(scores)
            diagnostics[s]={'per_seed':scores,'rmse':variables}
    passed=result['multi-complete']<=0.48 and result['single-complete']<=0.55
    return {'relative_rmse':result['multi-partial'] if passed else 1.0,'guard_pass':passed,
            'subset_scores':result,'details':diagnostics}
if __name__=='__main__':
    try:r=score(*sys.argv[1:])
    except Exception as e:r={'relative_rmse':float('inf'),'error':str(e)}
    print(json.dumps(r,allow_nan=True))
