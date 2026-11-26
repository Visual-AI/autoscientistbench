"""Literal SW score with diagnostic per-n median elapsed times."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json
import numpy as np
from validate import read_pred
from distance import sliced_wasserstein_distance


def score(pred,truth):
    with np.load(truth,allow_pickle=False) as f:
        ids=f['id'];ns=f['n'];ref=f['z_ref'];seeds=f['projection_seed']
    p=read_pred(pred,ids)
    distances=[];times=[]
    for j,i in enumerate(ids):
        z,t=p[str(i)];times.append(t)
        distances.append(sliced_wasserstein_distance(ref[j],z,n_proj=500,seed=int(seeds[j])))
    distances=np.asarray(distances);times=np.asarray(times)
    cells={}
    for n in sorted(set(ns)):
        mask=ns==n
        tm=float(np.median(times[mask]))
        cells[str(n)]={'sw':float(np.median(distances[mask])),'seconds':tm}
    raw=float(np.mean([v['sw'] for v in cells.values()]))
    return {'sw':raw,'sw_raw':raw,'per_n':cells}

if __name__=='__main__':
    try:out=score(*sys.argv[1:])
    except Exception as e:out={'sw':float('inf'),'error':str(e)}
    print(json.dumps(out))
