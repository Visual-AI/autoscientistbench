"""Per-instance adapter: configuration at train time, independent fits at predict time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from posterior import fit_tavie_student_and_sample_z


def sample_seed(X, y, seed):
    # Data-derived algorithm stream; does not expose the generator's hidden seeds.
    digest=hashlib.sha256(np.ascontiguousarray(X).tobytes()+np.ascontiguousarray(y).tobytes()).digest()
    return int.from_bytes(digest[:8], 'little') ^ int(seed)


def fit_sample(X, y, seed, config):
    return fit_tavie_student_and_sample_z(X,y,config['nu'],config['alpha'],config['n_samples'],seed=seed)


def worker(args):
    ident,X,y,seed,config=args
    t=time.perf_counter()
    z=fit_sample(X,y,seed,config)
    elapsed=time.perf_counter()-t
    return ident,z,elapsed


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    q=sp.add_parser('predict')
    for x in ['model','input','output']:q.add_argument('--'+x,required=True)
    a=p.parse_args()
    seed=int(os.environ['BENCH_SEED']);cpus=int(os.environ['BENCH_CPUS'])
    if a.cmd=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps({'nu':5,'alpha':1.,'n_samples':8000,'seed':seed,'cpus':cpus}))
        return
    cfg=json.loads((Path(a.model)/'config.json').read_text())
    with np.load(a.input,allow_pickle=False) as f:
        ids=f['id'];offsets=f['offsets'];X=f['X'];y=f['y']
    args=[(str(i),X[offsets[j]:offsets[j+1]],y[offsets[j]:offsets[j+1]],sample_seed(X[offsets[j]:offsets[j+1]],y[offsets[j]:offsets[j+1]],seed),cfg) for j,i in enumerate(ids)]
    with ProcessPoolExecutor(max_workers=min(cpus,len(args)),mp_context=mp.get_context('spawn')) as pool:
        result=list(pool.map(worker,args))
    with open(a.output,'wb') as f:
        np.savez(f,id=ids,z=np.stack([r[1] for r in result]),seconds=np.array([r[2] for r in result]))
    print(json.dumps({'instances':len(ids),'median_fit_sample_s':float(np.median([r[2] for r in result]))}))

if __name__=='__main__':main()
