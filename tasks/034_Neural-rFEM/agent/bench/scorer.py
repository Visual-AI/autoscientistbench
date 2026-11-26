#!/usr/bin/env python3
"""Frozen authors' conforming Q1 energy score. Usage: scorer.py PRED LABELS.
The zero prediction-time FEM declaration is checked; actual provenance requires patch audit.
FEM work here independently evaluates the submitted mesh.
"""
import os, sys, json
from pathlib import Path
# Small sparse systems: one CPU, independent of caller environment/cwd.
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mesh_io import read,matching,guards,PRED_COLUMNS,LABEL_COLUMNS,NODE_COLUMNS
METRIC='mean_relative_ritz_error'
def score(pred_path,label_path):
    pred=read(pred_path,PRED_COLUMNS);labels=read(label_path,LABEL_COLUMNS)
    matching(pred,labels);guards(pred)
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from fem_system import solve_and_loss_scipy
    values=[]
    for p,lab in zip(pred,labels):
        reference=lab['J_ref']
        if reference==0:raise ValueError('Zero reference energy')
        nodes=jnp.asarray([p[c] for c in NODE_COLUMNS],dtype=jnp.float64)
        sigma=jnp.asarray([lab['s1'],lab['s2']],dtype=jnp.float64).reshape(-1,2)
        energy=float(solve_and_loss_scipy(nodes,sigma)[0])
        values.append(np.sqrt(abs(reference-energy)/abs(reference)))
    values=np.asarray(values)
    if not np.isfinite(values).all():raise ValueError('Nonfinite FEM energy error')
    value=float(values.mean())
    return {METRIC:value,'score_rule_value':value,'all_admissible':True,
            'max_relative_error':float(values.max()),'scorer_fem_solves':len(values),
            'reported_train_s':pred[0]['train_s'],
            'reported_inference_max_s':max(r['inference_s'] for r in pred),
            'resource_evidence':'Zero prediction-time FEM declaration checked; actual offline/online provenance requires patch audit.'}
def main():
    try:out=score(sys.argv[1],sys.argv[2])
    except Exception as e:
        out={METRIC:float('inf'),'score_rule_value':1.0,'guard_failed':str(e)}
    print(json.dumps(out))
if __name__=='__main__':main()
