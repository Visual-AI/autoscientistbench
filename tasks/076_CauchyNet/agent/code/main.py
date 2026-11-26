"""Train the released configuration from scratch and export all ten independent models."""
import argparse, csv, json, multiprocessing as mp, os, random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from experiments.best_config_gap_filling import CauchyNet, train_one


def read(path):
    with open(path, newline='') as f: return list(csv.DictReader(f))


def count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)


def worker(seed, out, data):
    torch.set_num_threads(1)
    tx=read(Path(data)/'train/train.csv')
    vx=read(Path(data)/'val/input.csv'); vy={r['id']:r['y'] for r in read(Path(data)/'val/labels.csv')}
    x=torch.tensor([[float(r['x'])] for r in tx],dtype=torch.float32)
    y=torch.tensor([[float(r['y'])] for r in tx],dtype=torch.float32)
    xv=torch.tensor([[float(r['x'])] for r in vx],dtype=torch.float32)
    yv=torch.tensor([[float(vy[r['id']])] for r in vx],dtype=torch.float32)
    train=DataLoader(TensorDataset(x,y),batch_size=32,shuffle=True)
    val=DataLoader(TensorDataset(xv,yv),batch_size=32,shuffle=False)
    torch.manual_seed(seed); np.random.seed(seed)
    model=CauchyNet(hidden_size=64,r_re=2.5,r_im=0.4,seed=seed)
    ck=train_one(f'CauchyNet seed{seed}',model,train,val,epochs=3000,lr=0.05,imag_pen=0.0)
    ck.update(seed=seed,optimizer_steps=21000)
    torch.save(ck,Path(out)/f'state{seed}.pt')
    model.eval()
    exported=torch.jit.trace(model,x[:3],check_trace=True)
    # Serialization must preserve the fitted function, parameters and buffers.
    exported.save(str(Path(out)/f'model{seed}.ts'))
    result={'seed':seed,'params':count(model),'best_epoch':ck['best_epoch'],'train_time_s':ck['train_time_s']}
    print(json.dumps(result),flush=True)
    return result


def train(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if list(out.glob('model*.ts')): raise ValueError('output directory already contains models')
    torch.set_num_threads(1)
    seeds=list(range(10))
    # The metric fixes the model RNG seeds. BENCH_SEED controls only the order
    # in which independent jobs are submitted; it never changes those seeds.
    random.Random(int(os.environ.get('BENCH_SEED','1'))).shuffle(seeds)
    with ProcessPoolExecutor(max_workers=min(int(os.environ.get('BENCH_CPUS','8')),10),
                             mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(worker,s,str(out),os.environ['BENCH_DATA_DIR']) for s in seeds]
        rows=[f.result() for f in futures]
    (out/'model_card.json').write_text(json.dumps({'format':'ten-model-torchscript-v1',
        'execution_seed':int(os.environ.get('BENCH_SEED','1')),'models':sorted(rows,key=lambda r:r['seed'])},indent=2)+'\n')


def predict(a):
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','8')))
    rows=read(a.input)
    x=torch.tensor([[float(r['x'])] for r in rows],dtype=torch.float32)
    preds=[];params=[]
    for s in range(10):
        model=torch.jit.load(str(Path(a.model)/f'model{s}.ts'),map_location='cpu');model.eval()
        params.append(count(model))
        with torch.no_grad():
            result=model(x)
            if isinstance(result,tuple): result=result[0]
            preds.append(result.numpy().ravel())
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['id']+[f'pred_{s}' for s in range(10)]+[f'params_{s}' for s in range(10)])
        for i,r in enumerate(rows):w.writerow([r['id']]+[float(p[i]) for p in preds]+params)

if __name__=='__main__':
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    q=sp.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    a=p.parse_args();train(a) if a.cmd=='train' else predict(a)
