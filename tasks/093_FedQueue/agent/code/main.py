"""Protocol adapter for the archived FedQueue hard experiment."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from experiment.model_cnn import CNN
from partition import partition_indices

CODE = Path(__file__).resolve().parent
PROTOCOL = 'fedqueue-alpha01-rho09-v1'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def input_data(path):
    with np.load(path, allow_pickle=False) as z:
        return z['id'].copy(), torch.from_numpy(z['x'].copy()).unsqueeze(1).float().div(255)

def classes(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i+1000]).argmax(1) for i in range(0,len(x),1000)]).numpy()

def train(a):
    out = Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    data = Path(os.environ['BENCH_DATA_DIR']).resolve()
    seed = int(os.environ['BENCH_SEED'])
    config_dir = out/'configs'; config_dir.mkdir(exist_ok=True)
    logs=out/'process_logs';logs.mkdir(exist_ok=True)
    with np.load(data/'train/train.npz',allow_pickle=False) as z:
        parts=partition_indices(z['y'],seed)
        partition_ids={k:z['id'][v] for k,v in parts.items()}
    np.savez(out/'partitions.npz',**parts)
    np.savez(out/'partition_ids.npz',**partition_ids)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    uri=f'localhost:{port}'
    server=OmegaConf.load(CODE/'experiment/configs/server_nonIID_rho09.yaml')
    sc=server.server_configs; cc=server.client_configs
    sc.scheduler_kwargs.queue_seed=seed
    sc.scheduler_kwargs.fedqueue_log_dir=str(out)
    sc.logging_output_dirname=str(logs)
    sc.comm_configs.grpc_configs.server_uri=uri
    cc.model_configs.seed=seed
    for group,field in [('train_configs','loss_fn_path'),('train_configs','metric_path'),('model_configs','model_path')]:
        cc[group][field]=str(CODE/str(cc[group][field]).lstrip('./'))
    OmegaConf.save(server,config_dir/'server.yaml')
    configs=[]
    for client in range(1,5):
        cfg=OmegaConf.load(CODE/f'experiment/configs/client_{client}.yaml')
        cfg.comm_configs.grpc_configs.server_uri=uri
        cfg.train_configs.logging_output_dirname=str(logs)
        cfg.data_configs.dataset_path=str(CODE/'experiment/dataset_mnist.py')
        cfg.data_configs.dataset_kwargs.update(data_root=str(data),alpha=.1,seed=seed,partition_path=str(out/'partitions.npz'))
        p=config_dir/f'client_{client}.yaml';OmegaConf.save(cfg,p);configs.append(p)
    env=dict(os.environ,APPFL_HOME=str(out/'runtime/appfl'),PYTHONUNBUFFERED='1')
    Path(env['APPFL_HOME']).mkdir(parents=True,exist_ok=True,mode=0o700)
    # Match the reproduction's four-core envelope and one thread per process.
    cpus=sorted(os.sched_getaffinity(0))[:min(4,int(os.environ['BENCH_CPUS']))]
    processes=[];handles=[]
    def start(script,args,name):
        f=(logs/f'{name}.log').open('w');handles.append(f)
        p=subprocess.Popen(['taskset','-c',','.join(map(str,cpus)),sys.executable,str(CODE/'experiment'/script),*map(str,args)],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
        processes.append(p);return p
    try:
        sp=start('run_server.py',['--config',config_dir/'server.yaml','--results_dir',out],'server')
        deadline=time.monotonic()+600
        while True:
            if sp.poll() is not None:raise RuntimeError('server startup failed; see process_logs/server.log')
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:break
            if time.monotonic()>deadline:raise TimeoutError('server startup exceeded 600s')
            time.sleep(1)
        for client,cfg in enumerate(configs,1):start('run_client.py',['--config',cfg],f'client_{client}')
        print(f'seed={seed}: four clients started',flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('client/server failed; see process_logs')
            time.sleep(1)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in handles:f.close()
    # All training ends before validation selection, preserving scheduler timings.
    ids,x=input_data(data/'val/input.npz')
    with np.load(data/'val/labels.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['id']);y=z['y'].copy()
    model=CNN(num_channel=1,num_classes=10,num_pixel=28)
    curve=[];predictions=[]
    for r in range(1,51):
        path=out/f'checkpoints/round_{r:03d}.pt'
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        pred=classes(model,x);predictions.append(pred)
        curve.append(100.0*int(np.count_nonzero(pred==y))/len(y))
        print(f'validation round={r} accuracy={curve[-1]:.6f}',flush=True)
    selected=int(np.argmax(curve))+1
    model.load_state_dict(torch.load(out/f'checkpoints/round_{selected:03d}.pt',map_location='cpu',weights_only=True))
    model.eval()
    torch.jit.trace(model,torch.zeros(1,1,28,28)).save(str(out/'model.ts'))
    with (out/'logs/fedqueue_rounds.csv').open() as f:rows=list(csv.DictReader(f))
    steps=sum(int(r['Ek']) for r in rows)
    np.savez_compressed(out/'selection.npz',id=ids,pred=np.asarray(predictions))
    scheduler=OmegaConf.to_container(sc.scheduler_kwargs,resolve=True)
    scheduler.pop('queue_seed');scheduler.pop('fedqueue_log_dir')
    card={'format':'fedqueue-card/v1','protocol':PROTOCOL,'training_source':'train','seed':seed,'rounds':50,'clients':4,
          'alpha1':8,'alpha2':.1,'scheduler':scheduler,'total_local_steps':steps,
          'selected_round':selected,'val_accuracy':curve,'partition_counts':{k:len(v) for k,v in parts.items()},
          'model_sha256':sha(out/'model.ts'),'partitions_sha256':sha(out/'partition_ids.npz'),
          'rounds_sha256':sha(out/'logs/fedqueue_rounds.csv'),'queue_sha256':sha(out/'queue_draws.jsonl')}
    (out/'model_card.json').write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps({'selected_round':selected,'val_accuracy':curve[selected-1],'total_local_steps':steps}),flush=True)

def predict(a):
    model_dir=Path(a.model)
    card=json.loads((model_dir/'model_card.json').read_text())
    model=torch.jit.load(str(model_dir/'model.ts'),map_location='cpu')
    ids,x=input_data(a.input);pred=classes(model,x)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pred','total_local_steps','rounds','clients','protocol','selected_round'])
        for i,p in zip(ids,pred):w.writerow([int(i),int(p),card['total_local_steps'],card['rounds'],card['clients'],card['protocol'],card['selected_round']])

if __name__=='__main__':
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS','1')))
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    p=sp.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args();train(a) if a.cmd=='train' else predict(a)
