"""Offline final-epoch protocol for the released MLP experiment."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
import torch
from benchmarks.original.relu import CriticalReLUNet
from utils.schedules import get_dropout_schedule
from utils.training import train_epoch

CONFIG = dict(depth=6,width=256,input_dim=3072,output_dim=10,epochs=75,batch_size=100,
              lr=1e-4,weight_decay=1e-7,eta_min=1e-7,sigma_w_sq=1.98,sigma_b_sq=.02,
              optimizer='AdamW',loss='CrossEntropyLoss',dtype='float32')
DATA = None
OUT = None
RATES = None

def seed_group(seed):
    return list(range(42+25*(seed-1),67+25*(seed-1)))

def initialize(data_path, out, rates, threads):
    global DATA, OUT, RATES
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    z=np.load(data_path,allow_pickle=False)
    DATA=(torch.from_numpy(z['x']),torch.from_numpy(z['y'])) if 'y' in z else torch.from_numpy(z['x'])
    OUT=Path(out);RATES=rates

def model_for():
    return CriticalReLUNet(3072,256,10,RATES,1.98,.02)

def fit(seed):
    t=time.monotonic()
    torch.manual_seed(seed);np.random.seed(seed)
    model=model_for()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-7)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=75,eta_min=1e-7)
    crit=torch.nn.CrossEntropyLoss()
    for epoch in range(75):
        train_epoch(model,DATA,opt,crit,100)
        sched.step()
    name=f'model_{seed}.pt';path=OUT/name
    torch.save(model.state_dict(),path)
    row=dict(seed=seed,file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             epoch=75,optimizer_steps=75*((len(DATA[1])+99)//100))
    print(f'seed={seed} final epoch=75 wall={time.monotonic()-t:.2f}s',flush=True)
    return row

@torch.no_grad()
def infer(row):
    p=OUT/row['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    model=model_for();model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);model.eval()
    return np.concatenate([model(DATA[i:i+100]).numpy() for i in range(0,len(DATA),100)])

def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='mode',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    q=sp.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    a=p.parse_args();cpus=int(os.environ.get('BENCH_CPUS','16'));seed=int(os.environ.get('BENCH_SEED','1'))
    if cpus<4:raise ValueError('At least 4 CPUs required to retain released four-thread arithmetic')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    threads=4;workers=min(cpus//threads,25)
    if a.mode=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        rates=get_dropout_schedule('reverse_step',6,.1,.2)
        data=Path(os.environ['BENCH_DATA_DIR'])/'train/train.npz'
        with mp.get_context('spawn').Pool(workers,initializer=initialize,initargs=(str(data),str(out),rates,threads)) as pool:
            rows=pool.map(fit,seed_group(seed))
        card=dict(format='eoc-mlp-v1',config=CONFIG,bench_seed=seed,seeds=seed_group(seed),rates=[rates]*25,models=rows)
        (out/'model_card.json').write_text(json.dumps(card,indent=2)+'\n')
    else:
        card=json.loads((Path(a.model)/'model_card.json').read_text())
        # All members of this allocation experiment share the same dropout allocation.
        assert all(r==card['rates'][0] for r in card['rates'])
        assert seed==card['bench_seed']
        with mp.get_context('spawn').Pool(workers,initializer=initialize,initargs=(a.input,a.model,card['rates'][0],threads)) as pool:
            logits=np.stack(pool.map(infer,card['models']))
        with np.load(a.input,allow_pickle=False) as z: ids=z['ids']
        with open(a.output,'wb') as f:
            np.savez(f,ids=ids,logits=logits,seeds=np.asarray(card['seeds']),rates=np.asarray(card['rates']),
                     config=np.asarray(json.dumps(card['config'],sort_keys=True)),bench_seed=np.asarray(card['bench_seed']))
if __name__=='__main__':main()
