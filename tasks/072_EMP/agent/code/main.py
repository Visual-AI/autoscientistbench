"""Benchmark data, selection, serialization, and prediction entry points."""
import argparse,copy,importlib.util,json,os,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,TensorDataset
import emp_cli as ref

def frozen():
 path=Path(os.environ['BENCH_BENCH_DIR'])/'certificate.py'
 spec=importlib.util.spec_from_file_location('frozen_certificate',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class CachedCIFAR(Dataset):
 def __init__(self,path):
  with np.load(path,allow_pickle=False) as d:self.images=torch.from_numpy(d['x']);self.targets=torch.from_numpy(d['y'])
 def __len__(self):return len(self.targets)
 def __getitem__(self,i):
  x=self.images[i]
  top=torch.randint(0,9,size=(1,)).item();left=torch.randint(0,9,size=(1,)).item()
  x=x[:,top:top+32,left:left+32]
  if torch.rand(1)<0.5:x=x.flip(-1)
  return x,self.targets[i]
def save_npz(path,arrays):
 with open(path,'wb') as f:np.savez(f,**arrays)
def train(a):
 seed=int(os.environ.get('BENCH_SEED','1'));cpus=int(os.environ.get('BENCH_CPUS','32'))
 torch.set_num_threads(min(12,cpus));ref.set_seed(seed)
 Path('.tmp').mkdir(exist_ok=True);tempfile.tempdir='.tmp'
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=Path(os.environ['BENCH_DATA_DIR'])
 tr=DataLoader(CachedCIFAR(data/'train/train.npz'),batch_size=128,shuffle=True,num_workers=min(4,max(0,cpus-12)))
 v=frozen().load(data/'val/input.npz');lab=frozen().load(data/'val/labels.npz')
 va=DataLoader(TensorDataset(torch.from_numpy(v['x']),torch.from_numpy(lab['y'])),batch_size=128,shuffle=False,num_workers=0)
 model=ref.FCNet(3072,[1000,900,800,750,700,650,600,500,400,200,100,10])
 # The approved dense checkpoint is a fixed input, shipped read-only under data/dense: 200 epochs of CPU SGD are
 # bit-reproducible only on the host that froze the identity digest, so retraining cannot satisfy it. Prune this.
 dense=frozen().load(data/('dense/seed%d.npz'%seed))
 model.load_state_dict({k:torch.from_numpy(dense[k]) for k in model.state_dict()})
 pruned=copy.deepcopy(model)
 # Preserve the reproduced float32 EMP reduction setting and the inclusive threshold bug.
 torch.set_num_threads(min(4,cpus));sparsity=ref.EMP_global_magnitude(pruned,1.0);torch.set_num_threads(min(12,cpus))
 p={'seed':np.array(seed,dtype=np.int64)}
 for k,v in model.state_dict().items():p['dense::'+k]=v.detach().numpy().copy()
 for k,v in pruned.state_dict().items():
  p['pruned::'+k]=v.detach().numpy().copy()
  if k.endswith('.weight'):p['mask::'+k]=(v.detach().numpy()!=0)
 save_npz(out/'certificate.npz',p)
 card={'format':'fc12-mask/v1','certificate':'certificate.npz','seed':seed,'dense_sha256':frozen().dense_hash(p),'selected_epoch':int(dense['selected_epoch']),'sparsity':100*sparsity}
 (out/'model_card.json').write_text(json.dumps(card,indent=2));print(json.dumps(card),flush=True)
def predict(a):
 c=frozen();c.configure();p=c.load(Path(a.model)/'certificate.npz');x=c.load(a.input)
 p.update(id=x['id'],logits=c.forward(p,x['x'],'pruned::'));save_npz(a.output,p)
if __name__=='__main__':
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
 t=sp.add_parser('train');t.add_argument('--out',required=True)
 p=sp.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
 a=ap.parse_args()
 if hasattr(os,'sched_getaffinity'):
  os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:int(os.environ.get('BENCH_CPUS','32'))])
 train(a) if a.cmd=='train' else predict(a)
