"""Headless PAC adapter: frozen input partition, original per-cluster PCA."""
import argparse
import csv
import json
import os
import random
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


def fit_layout(packed, offsets):
    # Same calls and dtypes as the reproduction's common.pca2 / train_pac.
    # The method's internal random_state=42 is a fixed algorithm parameter.
    coords=[]
    for s,e in zip(offsets[:-1],offsets[1:]):
        X=np.unpackbits(packed[s:e],axis=-1,count=1024).astype(np.float32)
        coords.append(PCA(n_components=2,random_state=42).fit_transform(X).astype(np.float32))
    return np.vstack(coords)


def train(a):
    with np.load(Path(a.data)/'train/input.npz',allow_pickle=False) as z:
        ids,packed,offsets=[z[k] for k in ['ids','packed','offsets']]
    coords=fit_layout(packed,offsets)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'layout.npz',ids=ids,coords=coords)
    (out/'metadata.json').write_text(json.dumps({'seed':int(os.environ.get('BENCH_SEED',a.seed)),'pca_random_state':42,'cpus':int(os.environ.get('BENCH_CPUS','8'))})+'\n')


def predict(a):
    with np.load(Path(a.model)/'layout.npz',allow_pickle=False) as z:
        positions={i:n for n,i in enumerate(z['ids'])}; coords=z['coords']
    with np.load(a.input,allow_pickle=False) as z: ids=z['ids']
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['id','x','y'])
        for i in ids:
            x,y=coords[positions[i]]
            w.writerow([i,repr(float(x)),repr(float(y))])


if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train'); t.add_argument('--data',required=True); t.add_argument('--seed',type=int,default=1); t.add_argument('--out',required=True)
    q=sub.add_parser('predict'); q.add_argument('--model',required=True); q.add_argument('--input',required=True); q.add_argument('--output',required=True)
    a=p.parse_args(); seed=int(os.environ.get('BENCH_SEED','1')); random.seed(seed); np.random.seed(seed)
    train(a) if a.cmd=='train' else predict(a)
