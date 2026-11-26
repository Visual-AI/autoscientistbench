"""Power and fixed null constraints with full frozen betting-certificate replay.

Invalid interface -> Infinity (harness protocol failure); a valid betting method
violating either null cap -> 0, exactly the task's binding score rule.
"""
import json,math,os,sys,zipfile
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from multiprocessing import Pool
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_pred
import betting_interface as bi

def aggregate(reject,cell):
    rates={int(c):float(np.mean(reject[cell==c])) for c in np.unique(cell)}
    if set(rates)!=set(range(100)):raise ValueError('labels must cover the experimental grid')
    power=float(np.mean([v for c,v in rates.items() if c%10!=5]))
    null=[rates[i*10+5] for i in range(10)]
    pooled=float(np.mean(reject[cell%10==5]))
    ok=max(null)<=0.0695 and pooled<=0.0562
    return {'power':power if ok else 0.0,'power_raw':power,'null_by_n':null,'null_pooled':pooled,'null_max':max(null),'null_ok':ok}

def score(pred,truth):
    with np.load(truth,allow_pickle=False) as z:lab={k:z[k] for k in z.files}
    if zipfile.is_zipfile(pred):
        from private_betting import evaluate_artifact
        values,seed,execution=evaluate_artifact(pred,lab)
        result=aggregate(np.array([v[2] for v in values],dtype=np.int8),lab['cell'])
        result.update(interface_ok=True,replay_mismatches=0,seed=seed,execution=execution)
        return result
    # Original certificate tables remain supported for operator archive parity.
    rows,policy,seed=read_pred(pred)
    if 'BENCH_SEED' in os.environ and seed!=int(os.environ['BENCH_SEED']):
        raise ValueError('prediction seed differs from BENCH_SEED')
    ids=[str(i) for i in lab['id']]
    if len(ids)!=len(set(ids)) or set(ids)!=set(rows):raise ValueError('id mismatch')
    # Operator archive replay uses original seed schedule, unavailable in benchmark inputs.
    legacy=lab.get('legacy_seed')
    jobs=((lab['x'][k,:int(lab['n'][k])],iid,seed,policy) for k,iid in enumerate(ids))
    cpus=min(16,max(1,int(os.environ.get('BENCH_CPUS','16'))),len(os.sched_getaffinity(0)))
    bad=0;rejection=np.empty(len(ids),dtype=np.int8)
    with Pool(cpus) as pool:
        if legacy is None:
            results=pool.imap(bi.worker,jobs,chunksize=32)
        else:
            jobs=((lab['x'][k,:int(lab['n'][k])],iid,seed,policy,int(legacy[k])) for k,iid in enumerate(ids))
            results=pool.imap(archive_worker,jobs,chunksize=32)
        for k,(iid,(e,u,r)) in enumerate(results):
            pe,pu,pr=rows[iid]
            good=math.isclose(pe,e,rel_tol=1e-10,abs_tol=1e-12) and pu==u and pr==r
            if not good:
                return {'power':float('inf'),'interface_ok':False,'replay_mismatches':1,
                        'error':'reported capital/U/reject differs from frozen policy (stopped at first mismatch)','seed':seed}
            rejection[k]=r
    result=aggregate(rejection,lab['cell'])
    result.update(interface_ok=bad==0,replay_mismatches=int(bad),seed=seed)
    if bad:result['power']=float('inf')
    return result

def archive_worker(job):
    x,iid,seed,p,legacy=job
    return iid,bi.evaluate(x,iid,seed,p,legacy_seed=legacy)

if __name__=='__main__':
    try:
        if len(sys.argv)!=3:raise ValueError('usage: scorer.py PRED TRUTH')
        result=score(*sys.argv[1:])
    except Exception as e:result={'power':float('inf'),'interface_ok':False,'error':str(e)}
    print(json.dumps(result))
