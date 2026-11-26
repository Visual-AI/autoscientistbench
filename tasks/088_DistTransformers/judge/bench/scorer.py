"""Original float32 Monte Carlo KL, preserving samples and log Jacobian.

Timing fields are emitted by the frozen runner and checked by a replay probe;
patch audit is required because the harness has no automatic probe hook.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.distributions import InverseGamma
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction
from _dt_reference.distributions.distributions import GaussianMixtureModel
from _dt_reference.distributions.utils import decode_gmm_sample, kl_divergence


def score(pred,truth):
    ids,v=read_prediction(pred)
    with np.load(truth,allow_pickle=False) as z:
        lab={k:z[k] for k in z.files}
    if set(ids)!=set(lab['id'].tolist()) or len(ids)!=len(lab['id']):
        raise ValueError('id mismatch')
    index={iid:i for i,iid in enumerate(ids)}
    v=v[[index[iid] for iid in lab['id']]]
    valid=(v[:,[0,2,3,5,8,9]]>0).all() and np.allclose(v[:,0]+v[:,3],1,rtol=0,atol=2e-6)
    structural=bool((v[:,6]==2).all() and (v[:,7]==1).all())
    forward=float(v[:,8].mean());baseline=float(v[:,9].mean())
    if not valid or not structural:
        return dict(posterior_kl=float('inf'),guards_ok=False,valid_mixture=bool(valid),
                    components_and_amortisation_valid=structural,
                    forward_s=forward,baseline_s=baseline)
    values=[];draw_metrics=[]
    for draw in np.unique(lab['draw']):
        idx=np.flatnonzero(lab['draw']==draw)
        idx=idx[np.argsort(lab['position'][idx])]
        params=torch.tensor(v[idx,:6],dtype=torch.float32).reshape(-1,2,3)
        q=GaussianMixtureModel(**decode_gmm_sample(params,'scale_tril'))
        p=InverseGamma(torch.from_numpy(lab['a'][idx].copy()),torch.from_numpy(lab['b'][idx].copy()))
        with torch.no_grad(),torch.random.fork_rng():
            torch.set_rng_state(torch.from_numpy(lab['posterior_rng'][draw].copy()))
            kl=kl_divergence(p,q,torch.log,int(lab['n_mc']))
        if not torch.isfinite(kl).all():
            return dict(posterior_kl=float('inf'),guards_ok=False,reason='nonfinite KL')
        values.extend(kl.tolist());draw_metrics.append(float(kl.mean()))
    raw=float(np.mean(draw_metrics))
    return dict(posterior_kl=raw,guards_ok=True,draw_kl=draw_metrics,
                problem_sem=float(np.std(values,ddof=1)/np.sqrt(len(values))),
                draw_sem=float(np.std(draw_metrics,ddof=1)/np.sqrt(len(draw_metrics))) if len(draw_metrics)>1 else None,
                forward_s=forward,baseline_s=baseline,forward_ratio=forward/baseline,
                archived_baseline_forward_s=float(np.mean(lab['baseline_forward_s'])))


if __name__=='__main__':
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','8')))
    torch.set_num_interop_threads(1)
    try:
        print(json.dumps(score(*sys.argv[1:])))
    except Exception as e:
        print(json.dumps(dict(posterior_kl=float('inf'),guards_ok=False,reason=str(e))))
