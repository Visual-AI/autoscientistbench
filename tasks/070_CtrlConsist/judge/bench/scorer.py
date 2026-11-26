"""Released endpoint_idx=-1 CIR KL, averaged over five independent fits.
No statistical/model quality constraint is added to the unconstrained score rule.
"""
import json
import os
import sys
from pathlib import Path
# Bound trusted JAX runtime as well as candidate numerical libraries.
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:min(4,int(os.environ.get('BENCH_CPUS','4')))])
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from validate import read_prediction

def score(pred,labels):
    import jax.numpy as jnp
    with np.load(labels,allow_pickle=False) as z:
        ids,truth,sigma,repeat,dt=z['id'],z['drift'],z['sigma'],z['repeat'],float(z['dt'])
    d=read_prediction(pred,ids)
    values=[];errors=[]
    for i in range(5):
        mask=repeat==i
        if int(mask.sum()) != 1000:raise ValueError('invalid evaluation labels')
        # Keep released float32 reductions and order, including constant T/num_steps.
        diffs=jnp.asarray(truth[mask])-jnp.asarray(d[mask],dtype=jnp.float32)
        sigma_us=diffs/jnp.asarray(sigma[mask])
        per_traj=0.5*jnp.sum(sigma_us**2,axis=(-2,-1))*dt
        values.append(float(jnp.mean(per_traj)))
        errors.append(float(np.std(np.asarray(per_traj),ddof=1)/np.sqrt(len(per_traj))))
    value=sum(values)/len(values)
    if not np.isfinite(value):raise ValueError('nonfinite KL')
    return {'kl_to_solution':value,'per_repeat':values,'repeat_std':float(np.std(values)),
            'mc_standard_errors':errors,'format_ok':True}

if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'kl_to_solution':float('inf'),'format_ok':False,'reason':str(e)}
    print(json.dumps(result))
