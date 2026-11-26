"""Frozen FC12 certificate evaluator, independent of submitted code."""
import hashlib
import os
import numpy as np
import torch
import torch.nn.functional as F
DIMS=(3072,1000,900,800,750,700,650,600,500,400,200,100,10)
SHAPES={}
for i,(a,b) in enumerate(zip(DIMS[:-1],DIMS[1:])):
 SHAPES[f'net.{2*i}.weight']=(b,a)
 SHAPES[f'net.{2*i}.bias']=(b,)
WEIGHTS=[k for k in SHAPES if k.endswith('.weight')]
TOTAL=sum(int(np.prod(SHAPES[k])) for k in WEIGHTS)
MODEL_KEYS={'seed'}|{p+k for p in ('dense::','pruned::') for k in SHAPES}|{'mask::'+k for k in WEIGHTS}
PRED_KEYS=MODEL_KEYS|{'id','logits'}
def configure():
 torch.set_num_threads(min(12,int(os.environ.get('BENCH_CPUS',os.environ.get('OMP_NUM_THREADS','4')))))
def load(path):
 with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def dense_hash(p):
 h=hashlib.sha256()
 for k in sorted(SHAPES):
  a=p['dense::'+k]
  h.update(k.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes(order='C'))
 return h.hexdigest()
def validate_arrays(p,ids=None):
 assert set(p)==(PRED_KEYS if ids is not None else MODEL_KEYS),'certificate keys'
 assert p['seed'].shape==() and p['seed'].dtype==np.int64 and int(p['seed'])>=0,'seed'
 for prefix in ('dense::','pruned::'):
  for k,shape in SHAPES.items():
   a=p[prefix+k]; assert a.shape==shape and a.dtype==np.float32,'state shape/dtype'
   assert np.isfinite(a).all(),'nonfinite state'
 for k in WEIGHTS:
  a=p['mask::'+k]; assert a.shape==SHAPES[k] and a.dtype==np.bool_,'mask shape/dtype'
 if ids is not None:
  assert p['id'].dtype==np.int64 and p['id'].ndim==1,'id dtype/shape'
  assert len(np.unique(p['id']))==len(p['id']),'duplicate id'
  assert np.array_equal(p['id'],ids),'ids must equal input ids in order'
  assert p['logits'].shape==(len(ids),10) and p['logits'].dtype==np.float32,'logits shape/dtype'
  assert np.isfinite(p['logits']).all(),'nonfinite logits'
def mask_check(p):
 for k in SHAPES:
  dense,pruned=p['dense::'+k],p['pruned::'+k]
  if k in WEIGHTS:
   mask=p['mask::'+k]
   assert np.array_equal(pruned[mask].view(np.uint32),dense[mask].view(np.uint32)),'kept weights changed'
   assert np.all(pruned[~mask]==0),'removed weights nonzero'
  else: assert np.array_equal(pruned.view(np.uint32),dense.view(np.uint32)),'bias changed'
 return sum(int(p['mask::'+k].sum()) for k in WEIGHTS)
def authenticate(p,labels):
 seed=int(p['seed']); key=f'dense_hash_{seed}'
 assert key in labels,'unregistered dense seed'
 assert dense_hash(p)==str(labels[key]),'fixed dense checkpoint mismatch'
@torch.no_grad()
def forward(p,x,prefix):
 params={k:torch.from_numpy(p[prefix+k]) for k in SHAPES}
 result=[]
 for start in range(0,len(x),128):
  z=torch.from_numpy(np.ascontiguousarray(x[start:start+128])).flatten(1)
  for i in range(12):
   z=F.linear(z,params[f'net.{2*i}.weight'],params[f'net.{2*i}.bias'])
   if i<11: z.relu_()
  result.append(z.numpy())
 return np.concatenate(result)
def ce(logits,y):
 # Author evaluator: float32 mean CE per batch, weighted Python accumulation.
 total=0.
 for start in range(0,len(y),128):
  b=y[start:start+128]
  total+=float(F.cross_entropy(torch.from_numpy(logits[start:start+128]),torch.from_numpy(b)))*len(b)
 return total/len(y)
def verified_logits(p,x):
 actual=forward(p,x,'pruned::')
 assert np.allclose(actual,p['logits'],rtol=1e-5,atol=1e-4),'certificate does not reproduce logits'
 return actual
