#!/usr/bin/env python3
"""Frozen QuTiP simulation: mean instance worst-input hybrid infidelity.
Physical invalidity has the manifest's worst score 1.0; Infinity is the
harness disqualification sentinel, with that physical score in diagnostics.
"""
import os
# Full lint does not pass thread limits. Enforce the declared CPU ceiling here too.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]=str(min(8,max(1,int(os.environ.get(key,'8')))))
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pulse_io import instances, predictions
import pulse_core as core

def score(pred,labels):
    rows=instances(labels)
    pulses=predictions(pred,[r['id'] for r in rows])
    # Validate all submitted gate lists and costs before any expensive simulation.
    for gates in pulses.values():
        if core.validate(gates)>core.BUDGET:raise ValueError('CD amplitude exceeds pi/4')
    results=[]
    for r in rows:
        v=core.evaluate(pulses[r['id']],alpha=r['alpha'],theta=r['theta_t'])
        if not v['ncav600_check_passed']:raise ValueError('Ncav=600 convergence guard failed')
        results.append(v)
    vals=[v['worst_infidelity'] for v in results]
    return {'hybrid_infidelity':math.fsum(vals)/len(vals),'score_rule_value':math.fsum(vals)/len(vals),
            'guards_ok':True,'max_cutoff_difference':max(v['convergence_max_abs_difference'] for v in results),
            'max_cd_amplitude':max(v['total_cd_amplitude'] for v in results),
            'per_instance':[{ 'infidelity':v['infidelity_by_input'],'cutoff_difference':v['convergence_max_abs_difference']} for v in results]}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as exc:result={'hybrid_infidelity':float('inf'),'score_rule_value':1.0,'guards_ok':False,'error':str(exc)}
    print(json.dumps(result))
