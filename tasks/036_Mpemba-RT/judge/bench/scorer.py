"""Eq. (13): median log10 C across ten replicas, then mean across instances."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_predictions
from generator import REPLICAS

METRIC='median_log10_c'
# The benchmark requires Infinity for invalid submissions. Retain the manifest's
# finite failure score explicitly as a diagnostic (its printed approximation).
WORST=-0.897

def score(pred_path,labels_path):
    try:
        rows=read_predictions(pred_path,labels_path)
        costs={}; violations=[]
        with np.load(labels_path,allow_pickle=False) as z:
            for i,iid in enumerate(z['id']):
                iid=str(iid);n=int(z['n'][i]);p=z['p'][i,:n]
                weights=np.asfortranarray(z['modes'][i,:n])/z['pi'][i,:n,None]
                values=[]
                for rep in range(REPLICAS):
                    row=rows[(iid,rep)];perm=np.array(row['permutation'],dtype=np.int64)
                    if not np.array_equal(np.sort(perm),np.arange(n)):
                        violations.append(f'{iid}/{rep}: not a permutation');continue
                    state=p[perm]
                    if not np.array_equal(np.sort(state),np.sort(p)):
                        violations.append(f'{iid}/{rep}: probability multiset');continue
                    c=float(np.sum(np.abs(weights.T @ state)))
                    if not np.isfinite(c):raise ValueError('nonfinite recomputed cost')
                    values.append(c)
                costs[iid]=values
        if violations:return {METRIC:float('inf'),'score_rule_value':WORST,'violations':violations}
        # Represent a computed exact zero with the smallest float64 positive
        # value; otherwise the harness interprets -Infinity as guard failure.
        floor=np.nextafter(0.,1.)
        per={iid:float(np.median(np.log10(np.maximum(v,floor)))) for iid,v in costs.items()}
        result=float(np.mean(list(per.values())))
        return {METRIC:result,'score_rule_value':result,'per_instance':per,
                'per_instance_costs':costs,'permutation_check':True,
                'max_elapsed_s':max(r['elapsed_s'] for r in rows.values()),
                'zero_cost_count':sum(c==0 for v in costs.values() for c in v)}
    except Exception as e:
        return {METRIC:float('inf'),'score_rule_value':WORST,'violations':[str(e)]}

if __name__=='__main__':print(json.dumps(score(sys.argv[1],sys.argv[2])))
