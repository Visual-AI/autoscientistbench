"""Neural forward pass only. Q1 energy evaluation belongs to the scorer."""
import argparse, csv, hashlib, json, os, time
from pathlib import Path
# As in the reproduced mesh exporter, use one CPU for tiny forward passes.
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import jax.nn as nn
@jax.jit
def forward_pass(A_t,params):
    Ws,bs=params; H=A_t
    for i in range(len(Ws)-1):
        H=jnp.tanh(jnp.matmul(H,Ws[i])+bs[i])
    nodes_=jnp.squeeze(jnp.array([jnp.matmul(H,Ws[-1])+bs[-1]]))
    n_n=nodes_.size
    tx=nn.softmax(nodes_[:n_n//2]);ty=nn.softmax(nodes_[n_n//2:])
    nx=jnp.insert(jnp.insert(jnp.cumsum(tx),0,0.5),0,0).sort()
    ny=jnp.insert(jnp.insert(jnp.cumsum(ty),0,0.5),0,0).sort()
    return jnp.concatenate([nx,ny])
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    model=Path(a.model);card=json.loads((model/'model_card.json').read_text())
    assert hashlib.sha256((model/'trained_params.npz').read_bytes()).hexdigest()==card['weights_sha256']
    with np.load(model/'trained_params.npz',allow_pickle=False) as z:
        params=[[jnp.asarray(z[f'W{i}']) for i in range(3)],[jnp.asarray(z[f'b{i}']) for i in range(3)]]
    with open(a.input,newline='') as f: rows=list(csv.DictReader(f))
    compiled=forward_pass.lower(jnp.zeros(2,dtype=jnp.float64),params).compile()
    results=[]
    for row in rows:
        t0=time.perf_counter()
        mesh=np.asarray(compiled(jnp.asarray([float(row['s1']),float(row['s2'])]),params)).copy()
        for endpoint in (10,21):
            if abs(mesh[endpoint]-1.) <= 8*np.finfo(np.float64).eps:mesh[endpoint]=1.
        duration=time.perf_counter()-t0
        results.append([row['id'],*mesh,card['train_s'],duration,0])
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id']+[f'x{i}' for i in range(11)]+[f'y{i}' for i in range(11)]+['train_s','inference_s','fem_solves']);w.writerows(results)
    print(f'Wrote {len(results)} meshes',flush=True)
if __name__=='__main__':main()
