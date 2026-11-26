"""Frozen physical evaluator. Only diffusivity is supplied by the candidate."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_design
from frozen_propagators import propagate_density
from frozen_profiles import DensityComparator


TV_LIMIT={'test':0.0435,'val':0.18}

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

def score(pred,labels):
    with np.load(labels,allow_pickle=False) as z: lab=dict(z)
    d=read_design(pred,lab['id'])
    # PMF, periodic boundary, central initial condition and lag list are trusted.
    model=propagate_density(lab['positions'],d,lab['reduced_potential'],lab['lags'],periodic=True)
    comparator=DensityComparator(lab['positions'],periodic=True,normalize=True)
    tv=np.array([comparator.compare(ref,p).total_variation for ref,p in zip(lab['density'],model.densities,strict=True)])
    selected=np.isin(lab['lags'],[75,200])
    if selected.sum()!=2: raise ValueError('missing headline lags')
    raw=float(tv[selected].mean()); all_mean=float(tv.mean()); limit=TV_LIMIT[split_of(labels)]; ok=all_mean<=limit
    return {'tv':raw if ok else 1.0,'tv_raw':raw,'tv_all':all_mean,'tv_all_limit':limit,'quality_guard_ok':bool(ok),'pmf_input_rule_ok':True,'tv_by_lag':dict(zip(map(str,lab['lags']),map(float,tv)))}

if __name__=='__main__':
    try: result=score(sys.argv[1],sys.argv[2])
    except Exception as e: result={'tv':float('inf'),'invalid_design':str(e)}
    print(json.dumps(result))
