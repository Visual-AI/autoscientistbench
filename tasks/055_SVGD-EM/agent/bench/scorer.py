"""Ensemble test error at T=500: the feasible mean over trials, otherwise 1.0. Malformed output has Infinity (protocol rejection).
Errors are computed here from the submitted w and v with the frozen forward function of the authors' notebook.
Counts are declarations: actual oracle use additionally requires replay and audit.
"""
import json,os,sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import validate
METRIC='test_error'
WORST=1.0

def trial_metrics(w,v,X,y):
    import jax.numpy as jnp
    from bnn_metrics import log_pointwise_predrictive_density,test_error
    latent={'w':jnp.asarray(w),'v':jnp.asarray(v)}; images=jnp.asarray(X); labels=jnp.asarray(y).reshape(-1,1)
    return float(test_error(None,latent,images,labels)),float(log_pointwise_predrictive_density(None,latent,images,labels))

def _score(pred_path,labels_path):
    p=validate(pred_path,labels_path)
    with np.load(labels_path,allow_pickle=False) as z:
        X=z['X']; y=z['y']; base=z['baseline_lppd']; budget=z['baseline_budget']
    per=[trial_metrics(w,v,X,y) for w,v in zip(p['w'],p['v'])]
    err=np.array([e for e,_ in per]); lppd=np.array([l for _,l in per])
    limit=float(base.mean()-2*base.std(ddof=0))
    budget_ok=bool(np.all(p['oracle_counts']<=budget)); lppd_ok=bool(lppd.mean()>=limit)
    raw=float(err.mean()); feasible=budget_ok and lppd_ok
    return {METRIC:raw if feasible else WORST,'raw_test_error':raw,
        'trial_sd':float(err.std(ddof=0)),'trial_se':float(err.std(ddof=1)/np.sqrt(len(err))),
        'per_trial_test_error':err.tolist(),'lppd':float(lppd.mean()),'lppd_trial_sd':float(lppd.std(ddof=0)),
        'per_trial_lppd':lppd.tolist(),'lppd_limit':limit,
        'oracle_max_latent':float(p['oracle_counts'][:,:,0].max()),'oracle_max_theta':float(p['oracle_counts'][:,:,1].max()),
        'budget_ok':budget_ok,'lppd_ok':lppd_ok}
def score(pred_path,labels_path):
    try:return _score(pred_path,labels_path)
    except Exception as e:return {METRIC:float('inf'),'protocol_error':str(e)}

if __name__=='__main__':
    print(json.dumps(score(sys.argv[1],sys.argv[2])))
