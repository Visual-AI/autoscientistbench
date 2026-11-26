"""Protocol adapter: each collection uses the unchanged official JGS estimator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
import numpy as np
from src.jgs_estimator import joint_graph_sorting_estimate


def frozen_format():
    path = Path(os.environ['BENCH_BENCH_DIR'])/'format.py'
    spec = importlib.util.spec_from_file_location('_bench_format',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def work(job):
    i,graphs,config,seed = job
    np.random.seed(seed)
    random.seed(seed)
    start = time.perf_counter()
    result = joint_graph_sorting_estimate(graphs, k=config['k'], smoothing=config['smoothing'], target_size=config['target_size'])
    elapsed = time.perf_counter()-start
    return i,result['edge_frequencies'],elapsed


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    p=sp.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args();seed=int(os.environ.get('BENCH_SEED','1'));cpus=int(os.environ.get('BENCH_CPUS','1'))
    if a.cmd=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps(dict(k=None,smoothing=False,target_size=None,seed=seed)))
        print('JGS fits each observed collection at prediction time; configuration saved.')
        return
    config=json.loads((Path(a.model)/'config.json').read_text())
    fmt=frozen_format();obs=fmt.read_npz(a.input)
    def jobs():
        for i in range(len(obs['id'])):
            yield i,fmt.unpack_collection(obs,i),config,seed
    results=[]
    with ProcessPoolExecutor(max_workers=min(cpus,len(obs['id'])),mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,w,elapsed in pool.map(work,jobs(),chunksize=1):
            results.append((w,elapsed))
            if (i+1)%12==0:print('estimated',i+1,'collections',flush=True)
    ks=np.array([len(w) for w,_ in results],dtype=np.int64)
    offsets=np.concatenate(([0],np.cumsum(ks**2)))
    fmt.save_npz(a.output,id=obs['id'],k=ks,offsets=offsets,values=np.concatenate([w.ravel() for w,_ in results]),time_s=np.array([s for _,s in results]))

if __name__=='__main__':main()
