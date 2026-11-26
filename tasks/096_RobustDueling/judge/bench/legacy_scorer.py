"""Arithmetic mean of final cumulative dueling regret, independently reconstructed."""
import json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from legacy_validate import read_pred
from dueling_env import T,K


def score(pred_path,labels_path):
    pred=read_pred(pred_path)
    with np.load(labels_path,allow_pickle=False) as lab:
        ids=lab['ids'].tolist();utilities=lab['utilities'];h=int(lab['horizon'])
    if set(ids)!=set(pred) or len(set(ids))!=len(ids):raise ValueError('ID mismatch')
    if h!=T or utilities.shape!=(len(ids),T,K) or not np.isfinite(utilities).all():raise ValueError('Invalid labels')
    finals=[];max_error=0.
    for key,U in zip(ids,utilities):
        actions,n,reported=pred[key];pairs=np.asarray(actions,dtype=int)
        best=U.max(axis=1)
        increments=.5*((best-U[np.arange(T),pairs[:,0]])+(best-U[np.arange(T),pairs[:,1]]))
        final=float(np.cumsum(increments)[-1]);error=abs(final-reported);max_error=max(error,max_error)
        if not math.isclose(final,reported,rel_tol=1e-9,abs_tol=1e-7):raise ValueError('Reported regret differs from actions and hidden utilities')
        finals.append(final)
    return {'cumulative_regret':float(np.mean(finals)),'episode_std':float(np.std(finals,ddof=1)) if len(finals)>1 else 0.,'max_reconstruction_error':max_error,'per_episode':dict(zip(ids,finals)),'guard_ok':True}
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'cumulative_regret':float('inf'),'guard_ok':False,'reason':str(e)}
    print(json.dumps(result))
