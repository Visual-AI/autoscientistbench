#!/usr/bin/env python3
"""Mean 95% CI length, subject to coverage >= baseline coverage - 2 SE (baseline frozen in the labels)."""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_pred
Z=float(norm.ppf(.975))
def score(pred,labels):
    rows=read_pred(pred)
    with np.load(labels,allow_pickle=False) as z:
        ids=z['id'].astype(str).tolist();truth=z['m_true'];base=float(z['baseline_coverage'])
    if len(ids)!=len(set(ids)) or set(ids)!=set(rows):raise ValueError('id set mismatch')
    values=np.asarray([rows[i] for i in ids]);half=Z*values[:,1]*values[:,2]
    if not np.all(np.isfinite(half)):raise ValueError('interval overflow')
    length=2*half;coverage=float(np.mean(np.abs(values[:,0]-truth)<=half))
    m=len(ids)
    # Baseline-relative since 2026-11-09. Same 2 SE width as the nominal rule, re-centred on the baseline's own
    # frozen coverage: the band around .95 is centred on a coverage this baseline does not attain (.90625 on these
    # instances), which at M=32 left the smallest passing count 7% of one grid step above the threshold and
    # rejected 44% of submissions whose true coverage equals the baseline's. The SE floor keeps the band from
    # collapsing where the baseline covers every instance (val, 8/8). Reduces to the nominal rule at base=.95.
    tol=2*max(math.sqrt(base*(1-base)/m),math.sqrt(.0475/m))
    threshold=base-tol
    ok=coverage>=threshold
    return {'ci_length':float(length.mean()) if ok else float('inf'), 'ci_length_raw':float(length.mean()),
            'coverage':coverage,'coverage_threshold':threshold,'coverage_ok':ok,
            'baseline_coverage':base,'coverage_tolerance':tol,'nominal_threshold':float(.95-2*math.sqrt(.0475/m)),
            'mc_se':float(length.std(ddof=1)/np.sqrt(m)) if m>1 else 0.,
            'mean_abs_bias':float(np.mean(np.abs(values[:,0]-truth))),'n_instances':m}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'ci_length':float('inf'),'error':str(e)}
    print(json.dumps(result))
