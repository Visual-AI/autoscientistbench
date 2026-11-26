"""Observation-only entry points for the dynamic iVDFM experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
import numpy as np


def train(out):
    from omegaconf import OmegaConf
    root=Path(__file__).resolve().parent
    base=OmegaConf.load(root/'config/model/ivdfm.yaml')
    exp=OmegaConf.load(root/'config/experiment/factor_recovery.yaml')
    cfg=OmegaConf.merge(OmegaConf.create({'model':base,'dataset':{'r':5}}),
                        OmegaConf.create({'model':exp.suite_model_overrides.dynamic.ivdfm}))
    OmegaConf.resolve(cfg)
    # The original model uses the same seed for every simulation. Keep that
    # policy while making independent judge runs honor BENCH_SEED.
    cfg.model.seed=2025+int(os.environ.get('BENCH_SEED','1'))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(OmegaConf.to_container(cfg.model,resolve=True),indent=2)+'\n')


def fit_instance(args):
    iid,x,cfg=args
    import torch
    import pandas as pd
    from model.ivdfm import Model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed=int(cfg['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    start=time.monotonic()
    df=pd.DataFrame(x,columns=[f'series_{i}' for i in range(x.shape[1])])
    df.index=pd.RangeIndex(start=0,stop=len(df))
    model=Model(config_dict=cfg,device='cpu')
    model.fit(df,num_factors=5)
    factors=np.asarray(model.factors())
    if factors.shape!=(200,5) or not np.isfinite(factors).all():
        raise ValueError('fit did not return finite (200,5) factors')
    print(f'completed {iid} in {time.monotonic()-start:.3f}s',flush=True)
    return iid,factors


def predict(model,input_path,output):
    cfg=json.loads((Path(model)/'config.json').read_text())
    with np.load(input_path,allow_pickle=False) as z:
        if set(z.files)!={'id','x'}:raise ValueError('input must contain observations only')
        ids=z['id'];x=z['x']
    if x.shape!=(len(ids),200,20) or not np.isfinite(x).all():raise ValueError('bad observations')
    workers=min(max(1,int(os.environ.get('BENCH_CPUS','1'))),len(ids))
    # Each worker is a fresh process per instance: package state cannot couple fits.
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        results=list(pool.map(fit_instance,[(str(i),a,cfg) for i,a in zip(ids,x)]))
    with open(output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','t','f1','f2','f3','f4','f5'])
        for iid,factors in results:
            for t,row in enumerate(factors):w.writerow([iid,t,*[repr(float(v)) for v in row]])

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('train');a.add_argument('--out',required=True)
    a=s.add_parser('predict');a.add_argument('--model',required=True);a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='train':train(a.out)
    else:predict(a.model,a.input,a.output)
