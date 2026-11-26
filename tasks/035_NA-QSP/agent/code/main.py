#!/usr/bin/env python3
"""Protocol driver for the unchanged analytical GCR pulse (Eq. 30)."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random


def train(a):
    random.seed(a.seed)
    # Training-free baseline: save the public method configuration for prediction.
    config=json.loads((Path(a.data)/'train/config.json').read_text())
    config['seed']=a.seed
    Path(a.out).mkdir(parents=True,exist_ok=True)
    (Path(a.out)/'config.json').write_text(json.dumps(config)+'\n')


def predict(a):
    random.seed(int(os.environ.get('BENCH_SEED','1')))
    config=json.loads((Path(a.model)/'config.json').read_text())
    if config['method']!='GCR' or config['Delta']!=1:raise ValueError('unsupported baseline configuration')
    with open(a.input,newline='') as f:rows=list(csv.DictReader(f))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pulse','n_gates'])
        for r in rows:
            c=float(r['theta_t'])/(2*float(r['alpha']))
            # Rightmost momentum gate first, then position gate. No fitting.
            pulse=[{'type':'cd','lam':c,'phi':math.pi/2,'qubit_axis':[0,1,0]},
                   {'type':'cd','lam':c,'phi':0,'qubit_axis':[1,0,0]}]
            w.writerow([r['id'],json.dumps(pulse,separators=(',',':')),len(pulse)])

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='stage',required=True)
    t=sub.add_parser('train');t.add_argument('--out',required=True);t.add_argument('--data',required=True);t.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args();(train if a.stage=='train' else predict)(a)
