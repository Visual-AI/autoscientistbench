"""One online episode using the unchanged RCDP-UCB learning code."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
import torch
from runner import run_episode,load_env,initialize_worker
from contextual_dueling_bandit import DuelingGLMLearner
p=argparse.ArgumentParser()
for name in ('model','input','output'):p.add_argument('--'+name,required=True)
a=p.parse_args();json.loads((Path(a.model)/'model.json').read_text());row=json.loads(Path(a.input).read_text())
seed=int(os.environ.get('BENCH_SEED','1'));np.random.seed(seed);torch.manual_seed(seed);initialize_worker()
env=load_env();ep=env.make_episode(row['id'])
learner=DuelingGLMLearner(d=10,e=20,lambda_reg=1.,alpha=.1,C=25.,Lambda=10000.,mu_tau=100.,kappa=.25)
rec=run_episode(ep,learner,env.T)
with open(a.output,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['id','actions','rounds','regret']);w.writeheader();w.writerow(dict(id=row['id'],actions=json.dumps(rec['actions'],separators=(',',':')),rounds=rec['rounds'],regret=rec['regret']))
