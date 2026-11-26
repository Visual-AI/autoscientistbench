"""Protocol adapter: per-instance authors' factorization, with no shared fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent

def train(args):
    config=yaml.safe_load((HERE/'sample/factorize_tensor/standard_ttn.yml').read_text())
    # This is the completed hard-reproduction schedule (chi=4, all 100 sweeps).
    config['numerics']['fidelity']['max_bond_dimensions']=[4]
    config['numerics']['fidelity']['max_num_sweeps']=[100]
    config['algorithm_seed']=(int(os.environ.get('BENCH_SEED','1'))-1)%(2**32)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')

def threaded_tensordot(threads,min_size=2**18):
    """Split np.tensordot's permuting copies over threads. dot gets the same bytes and strides, so the same BLAS call."""
    from tensornetwork.backends.numpy.numpy_backend import NumPyBackend
    pool=ThreadPoolExecutor(max_workers=threads)
    plain,plain_conj=NumPyBackend.tensordot,NumPyBackend.conj
    def merges(x,axes):
        kept=[(x.shape[i],x.strides[i]) for i in axes if x.shape[i]>1]
        return all(s==n*t for (_,s),(n,t) in zip(kept,kept[1:]))
    def matrix(x,k):
        # x.reshape to 2-D on axis k; where that needs a copy, the same C-order copy made in threads
        shape=(int(np.prod(x.shape[:k],dtype=np.int64)),int(np.prod(x.shape[k:],dtype=np.int64)))
        if x.size<min_size or (merges(x,range(k)) and merges(x,range(k,x.ndim))):return x.reshape(shape)
        out=np.empty(x.shape,x.dtype)
        n=0;blocks=1
        while blocks<threads and n<x.ndim-1:blocks*=x.shape[n];n+=1
        list(pool.map(lambda i:np.copyto(out[i],x[i]),np.ndindex(x.shape[:n])))
        return out.reshape(shape)
    def tensordot(self,a,b,axes):
        if isinstance(axes,int) or max(a.size,b.size)<min_size or (len(axes[0])==a.ndim and len(axes[1])==b.ndim):return plain(self,a,b,axes)
        keep_a=[i for i in range(a.ndim) if i not in axes[0]];keep_b=[i for i in range(b.ndim) if i not in axes[1]]
        res=np.dot(matrix(a.transpose(keep_a+list(axes[0])),len(keep_a)),matrix(b.transpose(list(axes[1])+keep_b),len(axes[1])))
        return res.reshape([a.shape[i] for i in keep_a]+[b.shape[i] for i in keep_b])
    # conj of a real array is a plain copy, and nothing here writes to the large target
    NumPyBackend.conj=lambda self,x:x if x.size>=min_size and not np.iscomplexobj(x) else plain_conj(self,x)
    NumPyBackend.tensordot=tensordot

def solve(args):
    threaded_tensordot(int(os.environ.get('OMP_NUM_THREADS','1')))
    from ttnopt.factorize_tensor import factorize_tensor
    config=json.loads(Path(args.config).read_text())
    config.pop('algorithm_seed',None)
    config['target']=str(Path(args.target).resolve())
    config['output']['dir']=str(Path(args.out).resolve())
    path=Path(args.out).parent/'instance.yml';path.write_text(yaml.safe_dump(config))
    sys.argv=['ft',str(path)]
    factorize_tensor()

def predict(args):
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    cpus=min(int(os.environ.get('BENCH_CPUS','32')),len(os.sched_getaffinity(0)))
    threads=min(16,cpus);workers=max(1,cpus//threads)
    config=Path(args.model)/'config.json'
    with tempfile.TemporaryDirectory(prefix='ttn_',dir=out.parent) as temp:
        temp=Path(temp)
        with np.load(args.input,allow_pickle=False) as data:
            ids=data['ids'].copy()
            for i in range(len(ids)):
                d=temp/str(i);d.mkdir();np.save(d/'target.npy',data[f'target_{i}'])
        def one(i):
            d=temp/str(i)
            env=dict(os.environ)
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
                env[key]=str(threads)
            env['MPLCONFIGDIR']=str(d/'mpl')
            # BLAS workers sleep between calls instead of spinning, and freed arrays stay mapped instead of being zeroed again
            env.update(OPENBLAS_THREAD_TIMEOUT='8',MALLOC_MMAP_MAX_='0',MALLOC_TRIM_THRESHOLD_=str(2**36))
            command=[sys.executable,str(HERE/'bench_run.py'),'solve','--target',str(d/'target.npy'),'--config',str(config),'--out',str(d/'TTN')]
            start=time.monotonic()
            with (d/'factorize.log').open('w') as log:
                p=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=None)
            if p.returncode:
                print((d/'factorize.log').read_text()[-4000:],file=sys.stderr)
                raise RuntimeError(f'factorization failed: {p.returncode}')
            shutil.copy2(d/'factorize.log', out.parent/(out.name+f'.instance_{i}.log'))
            sweep_count=sum(line.startswith('Sweep count:') for line in (d/'factorize.log').read_text().splitlines())
            print(json.dumps({'instance':str(ids[i]),'wall_s':time.monotonic()-start,'cpus':threads,'sweeps':sweep_count}),flush=True)
            folder=d/'TTN/run1/tensors'
            record={f'edges_{i}':np.loadtxt(folder/'edges.dat',delimiter=',',dtype=np.int64),f'gauge_{i}':np.load(folder/'singular_values.npy')}
            record.update({f'tensor_{i}_{j}':np.load(folder/f'isometry{j}.npy') for j in range(22)})
            return record
        result={'ids':ids}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(one,range(len(ids))):result.update(r)
        with out.open('wb') as f:np.savez(f,**result)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--out',required=True)
    t=sub.add_parser('predict')
    for k in ['model','input','output']:t.add_argument('--'+k,required=True)
    t=sub.add_parser('solve')
    for k in ['target','config','out']:t.add_argument('--'+k,required=True)
    args=p.parse_args();globals()[args.mode](args)
