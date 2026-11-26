"""Protocol adapter: configuration artifact and independent trial process pool."""
import argparse, json, os, random, multiprocessing as mp, resource, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

CONFIG=dict(learning_rate=0.1,alpha=0.9,steps=500,particles=10,train_images=800,hidden_units=40,bandwidth=5.0)

def init_worker(config, seed):
    global COMPILED, CFG
    # One XLA thread per worker, as when each worker was bound to one CPU.
    os.environ['PJRT_NPROC']='1'
    import jax, jax.numpy as jnp, jax.random as jr
    from coinem.dataset import Dataset
    from bnn_model import BayesNN
    from algorithm import run_trial
    random.seed(seed); np.random.seed(seed)
    with np.load(Path(os.environ['BENCH_DATA_DIR'])/'train'/'train.npz',allow_pickle=False) as z:
        data=Dataset(jnp.array(z['X']),jnp.array(z['y']).reshape(-1,1))
    model=BayesNN(); CFG=config
    # The baseline consumes no algorithmic randomness. Changing this key leaves
    # its numerical result unchanged; candidate randomness uses BENCH_SEED.
    key=jr.PRNGKey(seed)
    COMPILED=jax.jit(lambda w,v,a,b: run_trial(model,data,{'w':w,'v':v},{'alpha':a,'beta':b},config,key,config['steps']))

def trial(args):
    ident,w0,v0,theta0=args; start=time.monotonic()
    latent,theta=COMPILED(w0,v0,theta0[0],theta0[1])
    w=np.asarray(latent['w']); v=np.asarray(latent['v']); t=np.array([theta['alpha'],theta['beta']],dtype=np.float32)
    from algorithm import oracle_counts
    return ident,w,v,t,oracle_counts(CFG,CFG['steps']),time.monotonic()-start,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train'); tr.add_argument('--out',required=True)
    pr=sub.add_parser('predict'); pr.add_argument('--model',required=True); pr.add_argument('--input',required=True); pr.add_argument('--output',required=True)
    a=ap.parse_args(); seed=int(os.environ.get('BENCH_SEED','1')); cpus=int(os.environ.get('BENCH_CPUS','8'))
    if a.mode=='train':
        Path(a.out).mkdir(parents=True,exist_ok=True)
        (Path(a.out)/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
        (Path(a.out)/'run.json').write_text(json.dumps(dict(seed=seed,cpus=cpus)))
        return
    config=json.loads((Path(a.model)/'config.json').read_text())
    with np.load(a.input,allow_pickle=False) as z:
        rows=list(zip(z['id'],z['w0'],z['v0'],z['theta0']))
    with ProcessPoolExecutor(max_workers=min(cpus,len(rows)),mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(config,seed)) as pool:
        results=list(pool.map(trial,rows))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'wb') as f:
        np.savez_compressed(f,id=np.array([r[0] for r in results]),w=np.stack([r[1] for r in results]),v=np.stack([r[2] for r in results]),theta=np.stack([r[3] for r in results]),oracle_counts=np.stack([r[4] for r in results]))
    print(json.dumps({'trials':len(results),'worker_seconds':[r[5] for r in results],'worker_peak_rss_mb':[r[6] for r in results]}),flush=True)
if __name__=='__main__':main()
