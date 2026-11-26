"""Protocol adapter; the published GA and HitRateModel remain unchanged."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import numpy as np
from model import HitRateModel
PARAMS=('rho','exp_mean','exp_std','artifact_freq','artifact_mean','artifact_std')
CODE=Path(__file__).resolve().parent

def train(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    outer=int(os.environ.get('BENCH_SEED',a.seed));cpus=int(os.environ.get('BENCH_CPUS','16'))
    # The metric fixes literal fit seeds 0–4. The outer seed controls only execution order.
    order=list(range(5));random.Random(outer).shuffle(order)
    for seed in order:
        cmd=[sys.executable,str(CODE/'optimize.py'),'--input_csv',str(Path(a.data).resolve()/'train/train.csv'),
             '--output',str(out/f'seed{seed}.json'),'--param_ranges',str(CODE/'config/param_ranges.json'),
             '--generations','100','--population_size','5000','--cxpb','0.4','--mutpb','0.4',
             '--n_cpus',str(cpus),'--seed',str(seed)]
        print(f'Fitting independent restart {seed}',flush=True)
        subprocess.run(cmd,check=True)
    (out/'run_manifest.json').write_text(json.dumps(dict(outer_seed=outer,fit_seeds=list(range(5)),execution_order=order,cpus=cpus,generations=100,population=5000,cxpb=.4,mutpb=.4,tournament_size=3),indent=2))

def predict(a):
    with open(a.input,newline='') as f: rows=list(csv.DictReader(f))
    fits={s:json.loads((Path(a.model)/f'seed{s}.json').read_text()) for s in range(5)}
    groups={}
    for i,r in enumerate(rows):groups.setdefault((int(r['run']),r['target']),[]).append(i)
    result=[None]*len(rows)
    for (s,t),indices in groups.items():
        params={k:fits[s][t][k] for k in PARAMS}
        x=np.array([float(rows[i]['pprop']) for i in indices]);d=np.array([float(rows[i]['definition']) for i in indices])
        pred=HitRateModel(params).get_pprop_hr(x,d)
        for i,v in zip(indices,pred):result[i]=[rows[i]['id'],repr(float(v)),*[repr(float(params[k])) for k in PARAMS]]
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pred',*PARAMS]);w.writerows(result)
    print(f'wrote {len(rows)} certified predictions')

if __name__=='__main__':
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True);t.add_argument('--data',required=True);t.add_argument('--seed',type=int,default=1)
    v=sp.add_parser('predict');v.add_argument('--model',required=True);v.add_argument('--input',required=True);v.add_argument('--output',required=True)
    a=p.parse_args();train(a) if a.command=='train' else predict(a)
