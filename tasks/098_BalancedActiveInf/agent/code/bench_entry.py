"""Protocol adapter for the frozen-prediction Table 1 cube-active experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'reproduced'))
sys.path.insert(0,os.environ['BENCH_BENCH_DIR'])


def train(out):
    from scipy.stats import norm
    import rpy2
    from rpy2.robjects import r
    import sampling_methods
    if sampling_methods.cube is None: raise RuntimeError('BalancedSampling is unavailable')
    provenance=json.loads((Path(os.environ['BENCH_DATA_DIR'])/'train/provenance.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    result=dict(stage='frozen_predictions',training='none',budget=.1,tau=.5,confidence=.9,
        seed=int(os.environ['BENCH_SEED']),provenance=provenance,
        numpy_path=np.__file__,rpy2_path=rpy2.__path__[0],R_home=str(r('R.home()')[0]),
        R_libraries=list(r('.libPaths()')),
        BalancedSampling_version=str(r('as.character(packageVersion("BalancedSampling"))')[0]))
    # predict starts once per draw, so it reads this instead of importing scipy (about 0.9 s per start)
    result['z']=float(norm.ppf(1.0-(1.0-result['confidence'])/2.0))
    source=Path(sys.prefix).parent
    for kind,path in [('source','code'),('data','data/raw'),('results','results')]:
        result['original_'+kind+'_accessible']=(source/path).exists()
        if result['original_'+kind+'_accessible']: raise RuntimeError('original research files mounted')
    (out/'model.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def predict(model,inp,output):
    from sampling_methods import cube_active_sampling,compute_sampling_probabilities
    from variance_estimators import estimate_cube_variance
    from env import load_episode,seed_draw
    config=json.loads((model/'model.json').read_text())
    meta,data,Oracle=load_episode(inp)
    if meta.get('source_scope',meta['scope'])=='smoke' and os.environ.get('BENCH_SMOKE')!='1': raise ValueError('explicit smoke required')
    if meta['scope']=='full' and meta['n_replications']!=10000: raise ValueError('full task requires 10000 draws')
    pred,u,err=(data[k] for k in ['y_pred','uncertainty','error_pred'])
    N=len(pred)
    pi=compute_sampling_probabilities(u,config['budget'],config['tau'])
    auxiliary=err.reshape(-1,1)
    z=config['z']
    records=[]
    seed_draw(meta)  # this baseline never samples in its own R; a method that does passes rpy2's r here
    for local_draw in range(meta['n_replications']):
        seed=int(meta.get('draw_index',local_draw))
        xi,pi_cube=cube_active_sampling(auxiliary,pi,random_state=seed)
        indices=np.flatnonzero(xi)
        oracle=Oracle()
        residuals=np.zeros(N,dtype=np.float64)
        residuals[indices]=oracle.query(indices)-pred[indices]
        estimate=(pred+residuals*xi/pi_cube).mean()
        variance=estimate_cube_variance(residuals,auxiliary,xi,pi_cube,N)
        margin=z*np.sqrt(variance)
        records.append(dict(id=local_draw,design='cube',selected=indices.tolist(),
            lower=float(estimate-margin),upper=float(estimate+margin),width=float(2.0*margin)))
        if meta['n_replications']>=1000 and (seed+1)%1000==0:
            print(f'cube-active: {seed+1}/{meta["n_replications"]}',flush=True)
    result=dict(version=1,predictions_sha256=meta['predictions_sha256'],
        designs={'cube':dict(dtype=str(pi.dtype),pi=pi.tolist())},trials=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(replications=len(records),population_size=N,r_seed=int(os.environ['BENCH_SEED']),
        design_budget=float(pi.sum()),mean_n_labeled=float(np.mean([len(x['selected']) for x in records])))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='stage',required=True)
    t=sub.add_parser('train'); t.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('predict')
    for flag in ['model','input','output']: q.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    if a.stage=='train': train(a.out)
    else: predict(a.model,a.input,a.output)
