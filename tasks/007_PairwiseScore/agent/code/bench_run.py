"""Independent fits on supplied comparison datasets, with isolated algorithm RNG."""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

def train(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    # Model directory is portable and self-contained; no shared training is used.
    shutil.copytree(Path(__file__).parent/'conf/experiment',out/'configs',dirs_exist_ok=True)
    subprocess.run(['g++','-O3','-march=native','-fno-math-errno','-ffp-contract=off','-fopenmp','-shared','-fPIC',str(Path(__file__).parent/'cpu_tau.cpp'),'-o',str(out/'cpu_tau.so')],check=True)
    (out/'config.json').write_text(json.dumps({'seed':int(os.environ['BENCH_SEED']),'method':'score-tau(x)','format':1}))

def worker(a):
    import numpy as np, torch
    from omegaconf import OmegaConf
    from cpu_main import run
    torch.set_num_threads(int(os.environ.get('BENCH_WORKER_THREADS','1')));torch.set_num_interop_threads(1)
    with np.load(a.input,allow_pickle=False) as z:
        D=z['dataset'].shape[1]
    name={2:'experiment2d',4:'experiment4d_uniform',6:'experiment6d',10:'experiment10d',16:'experiment16d'}[D]
    cfg=OmegaConf.load(Path(a.model)/'configs'/(name+'.yaml'))
    cfg.exp.seed=int(a.seed);cfg.exp.target='observed'
    os.environ['BENCH_INSTANCE']=str(Path(a.input).resolve())
    os.environ['REPRO_TAU_LIBRARY']=str(Path(a.model).resolve()/'cpu_tau.so')
    run(cfg,a.output)

def predict(a):
    import numpy as np
    model=Path(a.model).resolve()
    assert json.loads((model/'config.json').read_text())['seed']==int(os.environ['BENCH_SEED'])
    cpus=int(os.environ['BENCH_CPUS']);seed=int(os.environ['BENCH_SEED'])
    # Retain the released 4-thread large / 2-thread small CPU layout, with a shared thread budget.
    with tempfile.TemporaryDirectory(prefix='pairwise_fit_') as tmp:
        root=Path(tmp);jobs=[]
        with np.load(a.input,allow_pickle=False) as z:
            ids=z['ids'].tolist()
            for iid in ids:
                dest=root/iid;dest.mkdir()
                fields={k:z[iid+'__'+k] for k in ['dataset','bounds','means','covariance_matrix','sigma_data']}
                np.savez(dest/'input.npz',**fields)
                alg_seed=int.from_bytes(hashlib.sha256(f'{seed}:{iid}'.encode()).digest()[:4],'little')
                D=fields['dataset'].shape[1]
                jobs.append((D,iid,dest,alg_seed,min(cpus,4 if D>=10 else 2)))
        jobs.sort(reverse=True)
        active=[];pending=list(jobs);done=0
        try:
            while pending or active:
                while pending and sum(item[-1] for item in active)+pending[0][-1]<=cpus:
                    _,iid,dest,alg_seed,threads=pending.pop(0)
                    log=(dest/'run.log').open('w')
                    cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--input',str(dest/'input.npz'),'--model',str(model),'--output',str(dest),'--seed',str(alg_seed)]
                    env=os.environ.copy()
                    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BENCH_WORKER_THREADS']:env[key]=str(threads)
                    p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    active.append((p,log,iid,dest,threads))
                for item in active[:]:
                    p,log,iid,dest,_=item
                    if p.poll() is not None:
                        active.remove(item);log.close()
                        if p.returncode:raise RuntimeError(f'Instance {iid} failed: '+(dest/'run.log').read_text()[-6000:])
                        done+=1;print(f'completed {done}/{len(ids)}',flush=True)
                if active:time.sleep(.2)
            out={'ids':np.asarray(ids)}
            for iid in ids:out[iid]=np.load(root/iid/'samples.npy',allow_pickle=False)
            with open(a.output,'wb') as f:np.savez(f,**out)
        finally:
            for p,log,_,_,_ in active:
                if p.poll() is None:p.terminate()
            for p,log,_,_,_ in active:p.wait();log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
    t=s.add_parser('train');t.add_argument('--out',required=True)
    for name in ['predict','worker']:
        q=s.add_parser(name)
        for flag in ['model','input','output']:q.add_argument('--'+flag,required=True)
        if name=='worker':q.add_argument('--seed',required=True)
    a=p.parse_args();globals()[a.cmd](a)
