"""Protocol adapter; all learning equations and updates live in the original module."""
import argparse,csv,importlib.util,json,multiprocessing,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def load_env():
    path=Path(os.environ['BENCH_BENCH_DIR'])/'dueling_env.py'
    spec=importlib.util.spec_from_file_location('frozen_dueling_env',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def run_episode(episode, learner, horizon):
    """Original simulation ordering, with truth and delay queue behind the environment."""
    history={}
    for t in range(horizon):
        X=episode.get_context()
        a,b,omega=learner.select_arms(X,env=episode)
        ya,yb=episode.serve(a,b)
        learner.observe_context(X,a,b,ya,yb)
        if getattr(learner,'use_observed_feature_weighting',False):
            dz,omega=learner.record_observed_features(X,a,b,ya,yb)
        else:
            dz=getattr(learner,'last_delta_z',None)
            if omega is None: omega=1.
        history[t]=(X,a,b,ya,yb,omega,dz)
        for origin,outcome in episode.advance():
            x,aa,bb,y_a,y_b,w,delta=history.pop(origin)
            learner.current_delta_z=delta
            learner.update(x,aa,bb,outcome,y_a,y_b,w)
            learner.current_delta_z=None
    return episode.finish()


def worker(row):
    import numpy as np
    import torch
    from contextual_dueling_bandit import DuelingGLMLearner
    # Algorithm randomness is independent of environment seed and input ordering.
    seed=int(os.environ['BENCH_SEED'])
    np.random.seed(seed);torch.manual_seed(seed)
    env=load_env();ep=env.make_episode(int(row['seed']))
    learner=DuelingGLMLearner(d=10,e=20,lambda_reg=1.,alpha=.1,C=25.,Lambda=10000.,mu_tau=100.,kappa=.25)
    rec=run_episode(ep,learner,env.T)
    return {'id':row['id'],'actions':json.dumps(rec['actions'],separators=(',',':')),'rounds':rec['rounds'],'regret':rec['regret']}


def initialize_worker():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict']);p.add_argument('--out');p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    if a.stage=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        (out/'model.json').write_text(json.dumps({'method':'RCDP-UCB','seed':int(os.environ['BENCH_SEED']),'offline_training':False})+'\n');return
    json.loads((Path(a.model)/'model.json').read_text())
    with open(a.input,newline='') as f: rows=list(csv.DictReader(f))
    cpus=max(1,int(os.environ['BENCH_CPUS']))
    with ProcessPoolExecutor(max_workers=min(cpus,len(rows)),mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
        result=list(pool.map(worker,rows))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','actions','rounds','regret']);w.writeheader();w.writerows(result)
if __name__=='__main__':main()
