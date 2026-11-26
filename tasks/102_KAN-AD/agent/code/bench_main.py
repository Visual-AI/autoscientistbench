"""Split-safe entry points; five independent trials per external seed."""
import argparse,contextlib,hashlib,json,multiprocessing as mp,os,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torchinfo
from sklearn.preprocessing import StandardScaler
from kanad import KANAD

PARAMS=dict(window=96,order=2,batch_size=1024,epochs=100,lr=.01)
CACHE={}

def load_series(path):
    with np.load(path,allow_pickle=False) as z:
        ids=z['ids'].tolist();offsets=z['offsets'];values=z['values']
        return {ident:values[offsets[i]:offsets[i+1]].astype(str(z['dtypes'][i]) if 'dtypes' in z.files else values.dtype,copy=True) for i,ident in enumerate(ids)}

def difference(a):
    return (np.pad(a,((0,1)),'edge')-np.pad(a,((1,0)),'edge'))[:-1]

def fit(task):
    trial,ident,out,seed=task
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data,mean,scale=CACHE[ident];start=time.monotonic()
    with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
        method=KANAD(dict(PARAMS));method.audit_history=[]
        # This official statistics forward pass consumes RNG and updates BN: retain it.
        torchinfo.summary(method.model,(method.batch_size,method.window),verbose=0)
        method.train_valid_phase(data)
    torch.save(dict(state=method.model.state_dict(),mean=float(mean),scale=float(scale),params=PARAMS,seed=seed),Path(out)/f'{ident}_{trial}.pt')
    return dict(id=ident,trial=trial,seed=seed,epochs=len(method.audit_history),history=method.audit_history,wall_s=time.monotonic()-start)

def train(a):
    global CACHE
    torch.set_num_threads(1)
    data=Path(os.environ['BENCH_DATA_DIR']);train=load_series(data/'train/input.npz');valid=load_series(data/'val/input.npz')
    for ident,raw in train.items():
        tr=difference(raw);va=difference(valid[ident]);scaler=StandardScaler().fit(tr.reshape(-1,1))
        CACHE[ident]=(SimpleNamespace(train=scaler.transform(tr.reshape(-1,1)).flatten(),valid=scaler.transform(va.reshape(-1,1)).flatten()),scaler.mean_[0],scaler.scale_[0])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    base=int(os.environ.get('BENCH_SEED','1'))
    tasks=[(t,i,str(out),int.from_bytes(hashlib.sha256(f'{base}/{t}/{i}'.encode()).digest()[:4],'little')) for i in train for t in range(5)]
    tasks.sort(key=lambda t:len(CACHE[t[1]][0].train),reverse=True)
    workers=min(int(os.environ.get('BENCH_CPUS','1')),len(tasks))
    with mp.get_context('fork').Pool(workers) as pool,(out/'history.jsonl').open('w',buffering=1) as f:
        for n,rec in enumerate(pool.imap_unordered(fit,tasks),1):
            f.write(json.dumps(rec)+'\n')
            if n%50==0 or n==len(tasks):print(f'Completed {n}/{len(tasks)} fits',flush=True)
    (out/'metadata.json').write_text(json.dumps(dict(ids=list(train),trials=5,seed=base,params=PARAMS)))

def infer(task):
    trial,ident,model=task
    p=torch.load(Path(model)/f'{ident}_{trial}.pt',map_location='cpu',weights_only=False)
    # Keep sklearn's in-place dtype/rounding behavior when restoring the scaler.
    x=difference(CACHE[ident]);scaler=StandardScaler();scaler.mean_=np.array([p['mean']]);scaler.scale_=np.array([p['scale']]);scaler.var_=scaler.scale_**2;scaler.n_features_in_=1
    x=scaler.transform(x.reshape(-1,1)).flatten()
    with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet),contextlib.redirect_stderr(quiet):
        method=KANAD(dict(p['params']));method.model.load_state_dict(p['state']);method.test_phase(SimpleNamespace(test=x))
    return trial,ident,method.anomaly_score()

def predict(a):
    global CACHE
    torch.set_num_threads(1)
    # Prediction is deterministic, but honor the supplied algorithm seed for any initialization.
    seed=int(os.environ.get('BENCH_SEED','1'));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    CACHE=load_series(a.input);ids=list(CACHE);offsets=np.r_[0,np.cumsum([len(CACHE[i])-96 for i in ids])]
    scores=np.empty((5,int(offsets[-1])),dtype=np.float32);positions={i:n for n,i in enumerate(ids)}
    tasks=[(t,i,a.model) for i in ids for t in range(5)]
    workers=min(int(os.environ.get('BENCH_CPUS','1')),len(tasks))
    with mp.get_context('fork').Pool(workers) as pool:
        for t,i,s in pool.imap_unordered(infer,tasks):
            j=positions[i];scores[t,offsets[j]:offsets[j+1]]=s
    with open(a.output,'wb') as f:np.savez(f,ids=np.array(ids),offsets=offsets,scores=scores)
    print(f'Predicted {len(ids)} series, five trials',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    tr=sub.add_parser('train');tr.add_argument('--out',required=True)
    pr=sub.add_parser('predict');pr.add_argument('--model',required=True);pr.add_argument('--input',required=True);pr.add_argument('--output',required=True)
    a=p.parse_args();train(a) if a.stage=='train' else predict(a)
