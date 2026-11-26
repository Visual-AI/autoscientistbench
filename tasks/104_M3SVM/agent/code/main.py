"""Train ten independent affine classifiers; predict each run separately."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import csv
import numpy as np

RUNS=10

def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def worker(args):
    data_path, seed, run, out = args
    spec=importlib.util.spec_from_file_location('pmsvm',Path(__file__).with_name('main-opacus.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    data=np.load(data_path,allow_pickle=False)
    rng_seed=int(np.random.SeedSequence([seed,run]).generate_state(1)[0])
    card=mod.fit(data['X'],data['y'],rng_seed)
    card['run']=run
    Path(out,f'run_{run}.json').write_text(json.dumps(card))
    print(f'run={run} steps={card["steps"]} epsilon={card["epsilon"]}',flush=True)
    return card

def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(RUNS,int(os.environ.get('BENCH_CPUS','1'))), initializer=initialize_worker) as pool:
        cards=list(pool.map(worker,[(str(Path(args.data)/'train/train.npz'),args.seed,r,str(out)) for r in range(RUNS)]))
    (out/'model_card.json').write_text(json.dumps({'format':'dp-affine/v1','n_features':617,'n_classes':26,'runs':cards}))

def predict(args):
    import torch
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','1')))
    card=json.loads((Path(args.model)/'model_card.json').read_text())
    data=np.load(args.input,allow_pickle=False)
    X=torch.tensor(data['X'],dtype=torch.float32)
    predictions=[]
    for run in card['runs']:
        with torch.no_grad():
            p=torch.nn.functional.linear(X,torch.tensor(run['weight'],dtype=torch.float32),torch.tensor(run['bias'],dtype=torch.float32)).argmax(1).numpy()
        predictions.append(p)
    fields=['id']+[f'history_{r}' for r in range(RUNS)]+[f'delta_{r}' for r in range(RUNS)]+[f'pred_{r}' for r in range(RUNS)]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        meta=[json.dumps(r['history'],separators=(',',':')) for r in card['runs']]+[r['delta'] for r in card['runs']]
        for i,idx in enumerate(data['id']):w.writerow([int(idx)]+meta+[int(p[i]) for p in predictions])

if __name__=='__main__':
    p=argparse.ArgumentParser();subs=p.add_subparsers(dest='command',required=True)
    t=subs.add_parser('train');t.add_argument('--data',required=True);t.add_argument('--seed',type=int,required=True);t.add_argument('--out',required=True)
    t=subs.add_parser('predict');t.add_argument('--model',required=True);t.add_argument('--input',required=True);t.add_argument('--output',required=True)
    a=p.parse_args();globals()[a.command](a)
