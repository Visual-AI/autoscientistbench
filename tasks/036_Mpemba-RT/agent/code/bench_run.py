"""Protocol adapters around the unchanged Fig. 2 simulated annealer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, deque
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
from tools.notebook_utils import optimize_probability_permutation


def permutation_indices(original, state):
    bins=defaultdict(deque)
    for i,value in enumerate(original): bins[float(value)].append(i)
    return [bins[float(value)].popleft() for value in state]


def run_one(job):
    iid,rep,p,pi,modes,seed,iterations=job
    start=time.perf_counter()
    # Restore the original eigenvector array layout after padded NPZ loading.
    modes=np.asfortranarray(modes)
    state,history=optimize_probability_permutation(p,pi,modes,iterations=iterations,seed=seed)
    permutation=permutation_indices(p,state)
    return dict(id=iid,replica=rep,permutation=json.dumps(permutation,separators=(',',':')),
                elapsed_s=time.perf_counter()-start,iterations=iterations)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','predict'])
    ap.add_argument('--out');ap.add_argument('--model');ap.add_argument('--input');ap.add_argument('--output')
    a=ap.parse_args();seed=int(os.environ.get('BENCH_SEED','1'))
    if a.stage=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps({'iterations':50000,'seed':seed,'replicas':10})+'\n')
        return
    cfg=json.loads((Path(a.model)/'config.json').read_text())
    if cfg['seed']!=seed:raise ValueError('model and prediction BENCH_SEED disagree')
    jobs=[]
    with np.load(a.input,allow_pickle=False) as z:
        for i,iid in enumerate(z['id']):
            n=int(z['n'][i])
            for rep in range(10):
                # Seed 1 exactly reproduces the original ten seeds 0..9.
                alg_seed=10*(seed-1)+rep
                jobs.append((str(iid),rep,z['p'][i,:n],z['pi'][i,:n],z['modes'][i,:n],alg_seed,cfg['iterations']))
    cpus=max(1,int(os.environ.get('BENCH_CPUS','1')))
    with ProcessPoolExecutor(max_workers=min(cpus,len(jobs)),mp_context=mp.get_context('spawn')) as pool:
        rows=list(pool.map(run_one,jobs))
    with open(a.output,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','replica','permutation','elapsed_s','iterations'])
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({'completed':len(rows),'workers':min(cpus,len(jobs)), 'max_elapsed_s':max(r['elapsed_s'] for r in rows)}))

if __name__=='__main__':main()
