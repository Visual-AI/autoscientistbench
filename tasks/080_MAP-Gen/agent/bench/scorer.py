"""Bunny extrinsic COV, fixed histogram guards and authoritative projection."""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction
from mesh_projector import MeshConstraintProjector
from author_metrics import coverage

def fixed_metrics(a,b,hist_range):
    def probabilities(x):
        h,_=np.histogramdd(x,bins=50,range=hist_range)
        overflow=len(x)-int(h.sum())
        p=np.append(h.ravel(),overflow).astype(np.float64)+1e-12
        return p/p.sum(),overflow
    p,oa=probabilities(a);q,ob=probabilities(b);m=.5*(p+q)
    return {'fixed_jsd':float(.5*(np.sum(p*np.log(p/m))+np.sum(q*np.log(q/m)))),
            'fixed_tvd':float(.5*np.sum(np.abs(p-q))),'overflow':oa}

def measure(raw,labels):
    torch.set_num_threads(min(8,int(os.environ.get('BENCH_CPUS','8'))))
    projector=MeshConstraintProjector(str(Path(__file__).with_name('mesh.obj')),'cpu')
    with torch.no_grad(): projected=projector.project(torch.as_tensor(raw))[0].numpy()
    finite=np.isfinite(projected).all(1)
    cov=float(coverage(projected[finite],labels['reference'])) if finite.sum()>=5 else 0.
    return {'cov_raw':cov,'raw_finite':int(np.isfinite(raw).all(1).sum()),'projected_finite':int(finite.sum()),**fixed_metrics(projected,labels['reference'],labels['hist_range'])}

def score(pred,truth):
    with np.load(truth,allow_pickle=False) as z: labels={k:z[k] for k in z.files}
    _,raw=read_prediction(pred,labels['id'])
    result=measure(raw,labels)
    cap_j=float(labels['baseline_jsd'])+2*float(labels['baseline_jsd_sd'])
    cap_t=float(labels['baseline_tvd'])+2*float(labels['baseline_tvd_sd'])
    ok=(result['raw_finite']>=len(raw)-10 and result['projected_finite']>=len(raw)-10
        and result['fixed_jsd']<=cap_j+1e-12 and result['fixed_tvd']<=cap_t+1e-12)
    result.update(cov=result['cov_raw'] if ok else float('inf'),guards_ok=bool(ok),jsd_cap=cap_j,tvd_cap=cap_t)
    return result
if __name__=='__main__':
    try: result=score(*sys.argv[1:])
    except Exception as e: result={'cov':float('inf'),'guards_ok':False,'error':str(e)}
    print(json.dumps(result))
