#!/usr/bin/env python3
"""Certificate/trace probe. Not automatically invoked by the current harness.
Usage: check_model.py MODEL_DIR --input INPUT --pred PRED --data DATA [--seed S]
Run inside the isolated candidate execution environment. An observed trace is evidence, not a proof of executed training: audit the patch.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import os
import json
import math
import random
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_predictions,check_ids,PROTOCOL
from protocol import SCHEDULER,partition_indices

def require(ok,msg):
    if not ok:raise ValueError(msg)

def load_fixed_model(path):
    from model_def import CNN
    exported=torch.jit.load(str(path),map_location='cpu')
    state=exported.state_dict()
    require(all(torch.isfinite(v).all() for v in state.values()),'nonfinite fitted parameters')
    model=CNN(1,10,28).eval()
    model.load_state_dict(state,strict=True)
    return model

def check(a):
    root=Path(a.model);data=Path(a.data)
    card=json.loads((root/'model_card.json').read_text())
    require(card['format']=='fedqueue-card/v1','card format')
    require(card['protocol']==PROTOCOL,'protocol')
    require(card.get('training_source')=='train','training source must be train only; validation fitting is forbidden')
    require(card['rounds']==50 and card['clients']==4,'round/client protocol')
    require(card['alpha1']==8 and card['alpha2']==.1,'partition concentrations')
    for key in ('simulate_queue','queue_mode','queue_means','queue_sigma','slowdown'):
        require(card['scheduler'].get(key)==SCHEDULER[key],'queue environment '+key)
    if a.seed is not None:require(card['seed']==a.seed,'seed does not match judge')
    for field,path in [('model_sha256','model.ts'),('partitions_sha256','partition_ids.npz'),('rounds_sha256','logs/fedqueue_rounds.csv'),('queue_sha256','queue_draws.jsonl')]:
        require(hashlib.sha256((root/path).read_bytes()).hexdigest()==card[field],f'{field} mismatch')
    with (root/'logs/fedqueue_rounds.csv').open() as f:rows=list(csv.DictReader(f))
    steps=sum(int(r['Ek']) for r in rows)
    require(steps==card['total_local_steps'],'step recount mismatch')
    require(set(r['cid'] for r in rows)=={'Client1','Client2','Client3','Client4'},'client identities')
    require(all(int(r['Ek'])>=0 for r in rows),'negative local work')
    rounds=[json.loads(line)['round'] for line in (root/'round_history.jsonl').read_text().splitlines()]
    require(rounds[:50]==list(range(1,51)),'target checkpoint history')
    with np.load(data/'train/train.npz',allow_pickle=False) as z:
        train_ids=z['id'].copy();train_y=z['y'].copy()
    parts=partition_indices(train_y,card['seed'])
    with np.load(root/'partition_ids.npz',allow_pickle=False) as actual:
        require(set(actual.files)==set(parts),'partition keys')
        for k,v in parts.items():require(np.array_equal(actual[k],train_ids[v]),f'{k} partition')
    rng=random.Random(card['seed']);draws=0
    for line in (root/'queue_draws.jsonl').read_text().splitlines():
        r=json.loads(line);i=int(r['cid'].replace('Client',''))-1
        mu=math.log([1.5,2.5,3.5,4.5][i])-.5*.9*.9
        expected=max(0.0,rng.lognormvariate(mu,.9))
        require(abs(r['delay']-expected)<1e-12,'queue draw replay');draws+=1
    require(draws>0,'missing queue draws')
    with np.load(data/'val/labels.npz',allow_pickle=False) as z:
        limit=1.2*float(z['baseline_mean_total_local_steps'])
        require(math.isfinite(limit) and limit>0 and steps<=limit,'local-work budget')
        with np.load(root/'selection.npz',allow_pickle=False) as s:
            check_ids(s['id'],z['id']);require(s['pred'].shape==(50,len(z['id'])),'selection shape')
            curve=100*np.mean(s['pred']==z['y'][None,:],axis=1)
            selected_predictions=s['pred'][card['selected_round']-1].copy()
    require(np.allclose(curve,card['val_accuracy'],atol=1e-10,rtol=0),'validation curve')
    require(card['selected_round']==int(np.argmax(curve))+1,'validation-selected checkpoint')
    ids,p,meta=read_predictions(a.pred)
    for k in meta:require(meta[k]==card[k],f'prediction {k} differs from card')
    model=load_fixed_model(root/'model.ts')
    def infer(x):
        tx=torch.from_numpy(x)
        with torch.no_grad():return torch.cat([model(tx[i:i+1000]).argmax(1) for i in range(0,len(tx),1000)]).numpy()
    with np.load(data/'val/input.npz',allow_pickle=False) as z:
        vx=z['x'].astype(np.float32)[:,None]/255
    require(np.array_equal(infer(vx),selected_predictions),'model differs from selected validation predictions')
    with np.load(a.input,allow_pickle=False) as z:
        check_ids(ids,z['id']);x=z['x'].astype(np.float32)[:,None]/255
    require(np.array_equal(p,infer(x)),'predictions differ from model certificate')
    idx=np.arange(len(x)-1,-1,-1)[::2]
    require(np.array_equal(p[idx],infer(x[idx])),'subset predictions differ')
    return dict(guard_pass=True,total_local_steps=steps,queue_draws=draws,selected_round=card['selected_round'],prediction_equal=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model');ap.add_argument('--input',required=True);ap.add_argument('--pred',required=True);ap.add_argument('--data',required=True);ap.add_argument('--seed',type=int);ap.add_argument('--code');a=ap.parse_args()
    torch.set_num_threads(1)
    try:print(json.dumps(check(a)))
    except Exception as e:print(json.dumps({'guard_pass':False,'reason':str(e)}));sys.exit(1)
