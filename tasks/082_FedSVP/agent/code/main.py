"""Federated adaptation of offline frozen CLIP features; protocol entry points."""
import argparse,csv,json,os,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from utils.client import Client
from utils.server import Server

def configure():
    seed=int(os.environ.get('BENCH_SEED','1'))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,int(os.environ.get('BENCH_CPUS','16'))))
    return seed

def train(out):
    seed=configure();out.mkdir(parents=True,exist_ok=True)
    data=Path(os.environ['BENCH_DATA_DIR'])
    with np.load(data/'train/train.npz',allow_pickle=False) as z:
        ids=z['id'];x=torch.from_numpy(z['x']);y=torch.from_numpy(z['y']);membership=z['client'];rank=z['rank']
    with np.load(data/'text.npz',allow_pickle=False) as z:text=torch.from_numpy(z['text'])
    args=SimpleNamespace(num_shots=8,num_clients=10,dirichlet_alpha=.1,global_epochs=10,
        local_epochs=5,local_epochs_last=100,local_epochs_server=100,
        local_batch_size=8,global_batch_size=8,alpha=9.,beta=8.,
        local_lr=.0004,global_lr=.0004,lam_reg=.2,cosine_compute=False,use_dp=False)
    clients=[]
    for c in range(10):
        idx=np.where(membership==c)[0];idx=idx[np.argsort(rank[idx])]
        clients.append(Client(c,text,torch.empty((0,512)),torch.empty(0,dtype=torch.long),args,x[idx],y[idx],device=torch.device('cpu'),output_file=str(out/'training.txt')))
    # Trace counts are exported for audit. The method's upload is returned unchanged.
    uploads=[]
    for client in clients:
        original=client.update_prototypes
        def recorded(*a,_original=original,_cid=client.client_id,**kw):
            result=_original(*a,**kw)
            uploads.append({'client':_cid,'floats':result[0].numel(),'shape':list(result[0].shape)})
            return result
        client.update_prototypes=recorded
    server=Server(args,text,clients,str(out/'training.txt'));server.run()
    tensors={'global':server.global_semantic_prototypes.numpy(), 'text':text.numpy()}
    for c in clients:
        tensors[f'adapter_{c.client_id}']=c.prototypes.numpy()
        tensors[f'labels_{c.client_id}']=c.prototypes_labels.numpy()
        tensors[f'semantic_{c.client_id}']=c.global_semantic_prototypes.numpy()
    np.savez(out/'model.npz',**tensors)
    card={'format':'fedspa-feature-card/v1','encoder':'CLIP ViT-B/16','encoder_frozen':True,
          'classes':101,'dimension':512,'shots_pool':8,'clients':10,'rho':.1,'rounds':10,
          'seed':seed,'alpha':args.alpha,'beta':args.beta,'train_ids':ids.tolist(),
          'uploads':uploads,'state':'model.npz'}
    (out/'model_card.json').write_text(json.dumps(card,indent=2)+'\n')

def predict(model,input_path,output):
    configure();card=json.loads((model/'model_card.json').read_text())
    with np.load(input_path,allow_pickle=False) as z:ids=z['id'];x=torch.from_numpy(z['x'])
    with np.load(model/'model.npz',allow_pickle=False) as z:
        with torch.no_grad():
            g=(100.0*x@torch.from_numpy(z['global']).t()).topk(1,1,True,True)[1].flatten()
            heads=[]
            for c in range(10):
                affinity=x@torch.from_numpy(z[f'adapter_{c}']).t()
                cache_logits=((-1)*(card['beta']-card['beta']*affinity)).exp()@torch.from_numpy(z[f'labels_{c}'])
                clip_logits=100.0*x@torch.from_numpy(z[f'semantic_{c}']).t()
                heads.append((clip_logits+cache_logits*card['alpha']).topk(1,1,True,True)[1].flatten())
    preds=torch.stack([g]+heads,dim=1).numpy()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as f:
        w=csv.writer(f);w.writerow(['id','global']+[f'personal_{i}' for i in range(10)])
        w.writerows([[int(i)]+p.tolist() for i,p in zip(ids,preds)])

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','predict']);p.add_argument('--out',type=Path);p.add_argument('--model',type=Path);p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.mode=='train':train(a.out)
    else:predict(a.model,a.input,a.output)
if __name__=='__main__':main()
