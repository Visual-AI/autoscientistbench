#!/usr/bin/env python3
"""Original concentration MAE for analytic/ODE approximations. Provenance requires external audit."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_pred

def main(pred_path,labels_path):
    rows=read_pred(pred_path);pred={r['id']:r for r in rows}
    with open(labels_path,newline='') as f:lab=sorted(list(csv.DictReader(f)),key=lambda r:float(r['p']))
    if len(lab)!=len(pred) or set(pred)!={r['id'] for r in lab}:raise ValueError('label/prediction ids differ')
    y=np.array([float(r['y']) for r in lab]);cs=np.array([pred[r['id']]['pred'] for r in lab])
    if not np.isfinite(y).all():raise ValueError('invalid truth')
    raw=float(np.abs(cs-y).mean());cpu=rows[0].get('cpu_s');analytic=rows[0].get('analytic')
    se=float(lab[0]['baseline_mae_se'])
    result={'mae':raw,'mae_raw':raw,'declared_cpu_s':cpu,'declared_analytic':analytic,'reference_mc_se':se,'tie_absolute_delta':2*se,'external_audit_required':True}
    # A declaration is not proof of the implementation. An explicit out-of-scope
    # declaration is nevertheless invalid; omitted/true flags still need audit.
    if analytic == 0:
        result.update(mae=float('inf'),score_rule_value=0.5,reason='The task requires a deterministic analytic/ODE approximation, without simulation or simulation-fitted tables.')
    print(json.dumps(result))
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:print(json.dumps({'mae':float('inf'),'error':str(e)}))
