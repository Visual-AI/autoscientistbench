"""Eq.10: median of three mean log objectives; OOD guards are fixed by review."""
import json,os,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import validate
METRIC='normal_log_objective'
FAMILIES=['normal','cauchy','gumbel','laplace','lognormal']
THRESHOLDS={'cauchy':3.37,'gumbel':5.106,'laplace':4.988,'lognormal':5.028}

def aggregate_logs(logs):
    """Keep float32 torch reductions, including original median-of-seed-means."""
    v=torch.as_tensor(logs,dtype=torch.float32)
    return float(torch.quantile(v.mean(dim=1),.5))

def objective(x,c):
    return torch.square(torch.cdist(x,c)).min(dim=2).values.sum(dim=1)

def score(pred,labels):
    torch.set_num_threads(min(4,int(os.environ.get('BENCH_CPUS','4'))))
    ids,c,params,times=validate(pred,labels)
    with np.load(labels,allow_pickle=False) as z:
        x=z['x'];init=z['initial'];fam=z['family'];baseline=z['baseline_parameters']
    initial_ok=np.array_equal(c[:,:,0],np.broadcast_to(init,(3,*init.shape)))
    logs=[]
    for member in range(3):
        objs=[]
        for j in range(0,len(x),32):objs.append(objective(torch.from_numpy(x[j:j+32]),torch.from_numpy(c[member,j:j+32,-1])).log())
        logs.append(torch.cat(objs))
    logs=torch.stack(logs)
    if not bool(torch.isfinite(logs).all()):
        return {METRIC:float('inf'),'paper_score':6.343,'guards_ok':False,'reason':'nonfinite recomputed objective'}
    vals={f:aggregate_logs(logs[:,fam==f]) for f in FAMILIES}
    # Reported resource fields are only preliminary checks. Trusted replay is mandatory.
    ok=initial_ok and bool((params<=int(baseline)).all()) and all(vals[f]<=v for f,v in THRESHOLDS.items()) and all(np.isfinite(list(vals.values())))
    out={METRIC:vals['normal'] if ok else float('inf'),'paper_score':vals['normal'] if ok else 6.343,'guards_ok':bool(ok),'initial_centers_ok':bool(initial_ok),'families':vals,'ood_thresholds':THRESHOLDS,'parameters_reported':params.tolist(),'inference_s_reported':times.tolist(),'resource_probe_required':True,'per_member':{f:logs[:,fam==f].mean(dim=1).tolist() for f in FAMILIES},'task_se':{f:(logs[:,fam==f].std(dim=1)/np.sqrt(sum(fam==f))).tolist() for f in FAMILIES}}
    return out
if __name__=='__main__':
    try:print(json.dumps(score(*sys.argv[1:])))
    except Exception as e:print(json.dumps({METRIC:float('inf'),'paper_score':6.343,'guards_ok':False,'reason':str(e)}))
