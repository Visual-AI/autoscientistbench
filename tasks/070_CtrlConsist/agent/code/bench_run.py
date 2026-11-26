"""Protocol adapter: five independent CIR fits, then pointwise drift prediction."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

def build(cfg):
    from problems import PROBLEMS
    from src.consistency_bridge import ConsistencyBridge
    from train_bridge import _build_model
    p = PROBLEMS[cfg['problem']['name']](cfg['problem'])
    b = ConsistencyBridge(shape=p['shape'], x_0=p['x_0'], x_T=p['x_T'],
        base_drift_fn=p['base_drift'], sigma_fn=p['sigma_fn'],
        model=_build_model(cfg['model'],p['shape'][0]), bridge_config=cfg['bridge'],T=p['T'])
    return b

def worker(a):
    import numpy as np
    import yaml
    if a.action == 'fit':
        from train_bridge import run_training
        cfg = yaml.safe_load((Path(os.environ['BENCH_DATA_DIR'])/'train/config.yaml').read_text())
        cfg['train']['seed'] = a.seed
        run_training(cfg, str(Path(a.model)/f'repeat{a.repeat}'))
    else:
        import pickle
        import jax.numpy as jnp
        from src.evaluation import batch_drift_fn
        d = Path(a.model)/f'repeat{a.repeat}'
        cfg = yaml.safe_load((d/'config.yaml').read_text())
        with (d/'checkpoint.pkl').open('rb') as f:
            params = pickle.load(f)['ema_params']
        bridge = build(cfg)
        with np.load(a.input, allow_pickle=False) as z:
            mask = z['repeat'] == a.repeat
            ids, xs, ts = z['id'][mask], z['x'][mask], z['t']
        drift = batch_drift_fn(lambda x,t: bridge.controlled_drift(params,x,t), jnp.asarray(xs), jnp.asarray(ts))
        with open(a.output,'wb') as f:
            np.savez(f,id=ids,drift=np.asarray(drift[:,:-1]))

def run(a):
    cpus = sorted(os.sched_getaffinity(0))[:max(1,int(os.environ.get('BENCH_CPUS','16')))]
    n = min(5,len(cpus))
    groups = [cpus[i*len(cpus)//n:(i+1)*len(cpus)//n] for i in range(n)]
    seed = int(os.environ.get('BENCH_SEED','1'))
    if a.action == 'train':
        Path(a.out).mkdir(parents=True,exist_ok=True)
        (Path(a.out)/'config.json').write_text(json.dumps({'bench_seed':seed,'training_seeds':[5*(seed-1)+i for i in range(5)]}))
    else:
        cfg=json.loads((Path(a.model)/'config.json').read_text())
        if cfg['bench_seed'] != seed: raise ValueError('BENCH_SEED must match the trained model')
    with tempfile.TemporaryDirectory(prefix='cir_predict_') as tmp:
        def one(i):
            group=groups[i%n]
            env=os.environ.copy()
            for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
                env[name]=str(len(group))
            # XLA takes its thread pool size from PJRT_NPROC, as it did from the bound group.
            env['PJRT_NPROC']=str(len(group))
            cmd=[sys.executable,str(HERE/'bench_run.py')]
            if a.action=='train':
                cmd+=['fit','--model',a.out,'--repeat',str(i),'--seed',str(5*(seed-1)+i)]
                logpath=Path(a.out)/f'repeat{i}.log'
            else:
                cmd+=['infer','--model',a.model,'--repeat',str(i),'--input',a.input,'--output',str(Path(tmp)/f'{i}.npz')]
                logpath=Path(tmp)/f'{i}.log'
            with logpath.open('w') as log:
                r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode: raise RuntimeError(logpath.read_text()[-6000:])
        # Fixed lane queues keep the thread budgets within BENCH_CPUS even when fewer than five CPUs are available.
        def lane(j):
            for i in range(j,5,n): one(i)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lane,range(n)))
        if a.action=='predict':
            import numpy as np
            arrays=[]
            for i in range(5):
                with np.load(Path(tmp)/f'{i}.npz',allow_pickle=False) as z:
                    arrays.append((z['id'],z['drift']))
            with open(a.output,'wb') as f:
                np.savez(f,id=np.concatenate([v[0] for v in arrays]),drift=np.concatenate([v[1] for v in arrays]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','predict','fit','infer'])
    for k in ('out','model','input','output'):p.add_argument('--'+k)
    p.add_argument('--repeat',type=int);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.action in ('fit','infer'):worker(a)
    else:run(a)
