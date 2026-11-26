"""Worker-side public predictions and accounted RPC label access; no oracle payload."""
import hashlib,json
import random
from pathlib import Path
import numpy as np
from private_rpc import call
def seed_draw(meta,r=None):
 """Initialize a fresh policy draw; baseline NumPy keeps its original loop seed."""
 np.random.seed(int(meta['draw_index']))
 random.seed(int(meta['draw_seed']))
 if r is not None:r['set.seed'](int(meta['draw_seed']))

def load_episode(path):
 root=Path(path);meta=json.loads((root/'manifest.json').read_text())
 if meta.get('protocol')!='private-active-label-v2' or meta.get('n_replications')!=1:raise ValueError('one isolated draw required')
 if (root/'oracle.npy').exists():raise ValueError('oracle payload must not be public')
 if hashlib.sha256((root/'predictions.npz').read_bytes()).hexdigest()!=meta['predictions_sha256']:raise ValueError('frozen predictions changed')
 with np.load(root/'predictions.npz',allow_pickle=False) as z:predictions={k:z[k].copy() for k in ('y_pred','uncertainty','error_pred')}
 class Oracle:
  def __init__(self):self.seen=set()
  def query(self,indices):
   a=np.asarray(indices)
   if a.ndim!=1 or a.dtype.kind not in 'iu' or len(set(a.tolist()))!=len(a):raise ValueError('integer indices required')
   ids=[int(i) for i in a]
   values=np.asarray(call('query',ids),dtype=np.float64)
   self.seen.update(ids);return values
 return meta,predictions,Oracle
