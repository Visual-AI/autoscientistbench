"""Recompute the official metric from complete actions and trusted hidden labels."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_validate import read_pred
from private_clasp import T, _round_metrics


def score(pred_path, labels_path):
    rows=read_pred(pred_path)
    with np.load(labels_path,allow_pickle=False) as z:lab=dict(z)
    if [r['id'] for r in rows]!=list(lab['ids']):
        raise ValueError('label/prediction IDs differ')
    totals=[]; consistent=True; domain=True
    for i,r in enumerate(rows):
        actions=np.asarray(r['actions'],dtype=float)
        domain=domain and bool(np.all(actions>=-1e-6) and np.all(actions<=1+1e-6))
        total=np.zeros(3)
        for t in range(T):
            data=tuple(lab[k][i,t] for k in ('H','y','A','b'))
            total+=_round_metrics(data, actions[t].reshape(-1,1))
        consistent=consistent and bool(np.allclose(total,[r['loss'],r['ccv1'],r['ccv2']],rtol=1e-8,atol=1e-8))
        totals.append(total)
    mean=np.mean(totals,axis=0)
    loss_cap=1.01*float(np.mean(lab['baseline_loss']))
    ccv1_cap=1.05*float(np.mean(lab['baseline_ccv1']))
    guard_loss=bool(mean[0]<=loss_cap);guard_ccv1=bool(mean[1]<=ccv1_cap)
    ok=guard_loss and guard_ccv1 and consistent and domain
    return dict(ccv2=float(mean[2]) if ok else float('inf'),mean_loss=float(mean[0]),mean_ccv1=float(mean[1]),
                raw_ccv2=float(mean[2]),loss_cap=loss_cap,ccv1_cap=ccv1_cap,guard_loss=guard_loss,
                guard_ccv1=guard_ccv1,guard_reported_metrics=consistent,guard_domain=domain)

if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except (ValueError,KeyError,TypeError,OSError,IndexError) as e:result={'ccv2':float('inf'),'error':str(e)}
    print(json.dumps(result))
