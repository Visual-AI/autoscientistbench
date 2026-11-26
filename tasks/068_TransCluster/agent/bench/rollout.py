"""Frozen prediction replay. Adapters expose load_model and one step under the fixed 20-update protocol."""
import argparse,concurrent.futures,importlib.util,json,os,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch

def adapter(code):
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('candidate_adapter',code/'bench_adapter.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

BASELINE_PARAMETERS=21672

def count_parameters(api,model,model_dir,index):
    if isinstance(model,torch.nn.Module):
        count=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    else:
        count=0
        with np.load(Path(model_dir)/f'model{index}.parameters.npz',allow_pickle=False) as archive:
            for name in archive.files:
                value=archive[name]
                if not np.issubdtype(value.dtype,np.number) or not np.isfinite(value).all():raise ValueError('invalid learned parameter array')
                count += value.size*(2 if np.iscomplexobj(value) else 1)
    if count>BASELINE_PARAMETERS:raise ValueError('fitted-model capacity budget exceeded')
    if hasattr(api,'parameter_count') and int(api.parameter_count(model))!=count:raise ValueError('false parameter declaration')
    return count

def twenty_steps(api,model,samples,initial):
    if not callable(getattr(api,'step',None)):raise ValueError('adapter must expose one clustering step, not an unbounded solve')
    centers=initial.clone();trajectory=[centers.clone()];elapsed=0.
    for _ in range(20):
        start=time.perf_counter();centers=torch.as_tensor(api.step(model,samples,centers),dtype=torch.float32);elapsed+=time.perf_counter()-start
        if centers.shape!=initial.shape or not torch.isfinite(centers).all():raise ValueError('step must return finite centers with unchanged shape')
        trajectory.append(centers.clone())
    return torch.stack(trajectory,dim=1),elapsed

def worker(code,model,inp,out,seed,index,threads):
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    clock=time.perf_counter
    a=adapter(code)
    with np.load(inp,allow_pickle=False) as z:x=torch.from_numpy(z['x']);initial=torch.from_numpy(z['initial'])
    allc=[];elapsed=0.;nparams=0
    with torch.no_grad():
        for j in range(0,len(x),32):
            # The released evaluator re-instantiates the model at every batch.
            m=a.load_model(model,index)
            nparams=max(nparams,count_parameters(a,m,model,index))
            trajectory,duration=twenty_steps(a,m,x[j:j+32],initial[j:j+32])
            elapsed+=duration;allc.append(trajectory)
    with open(out,'wb') as f:np.savez(f,centers=torch.cat(allc).numpy(),parameters=np.array(nparams),inference_s=np.array(elapsed))

def run(code,model,inp,out,seed,cpus):
    threads=min(16,max(1,cpus//3));out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='rollout_',dir=out.parent) as temp:
        def one(index):
            p=Path(temp)/f'member{index}.npz';env=dict(os.environ)
            for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]=str(threads)
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--code',str(code),'--model',str(model),'--input',str(inp),'--output',str(p),'--seed',str(seed+104729*index),'--index',str(index),'--threads',str(threads)],env=env,check=True)
            with np.load(p) as z:return {k:z[k] for k in z.files}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,cpus)) as pool:members=list(pool.map(one,range(3)))
        with np.load(inp,allow_pickle=False) as z:ids=z['id']
        with open(out,'wb') as f:np.savez(f,id=ids,centers=np.stack([v['centers'] for v in members]),parameters=np.array([int(v['parameters']) for v in members]),inference_s=np.array([float(v['inference_s']) for v in members]))
    print(json.dumps({'parameters':[int(v['parameters']) for v in members],'inference_s':[float(v['inference_s']) for v in members]}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--code',type=Path);p.add_argument('--model',type=Path);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);p.add_argument('--seed',type=int);p.add_argument('--index',type=int);p.add_argument('--threads',type=int)
    a=p.parse_args();worker(a.code,a.model,a.input,a.output,a.seed,a.index,a.threads)
