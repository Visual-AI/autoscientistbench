"""Final submitted-model mean member CE reduction against the fixed Constant arm.
Artifact integrity and training provenance additionally require the probe / patch audit.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from schema import load,guard

def score(pred,labels):
    d=load(pred);guard(d)
    with np.load(labels,allow_pickle=False) as z:
        ids=z['ids'];y=z['y']
    if not np.array_equal(d['ids'],ids) or y.shape!=(len(ids),) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=10):raise ValueError('invalid labels / IDs')
    torch.set_num_threads(min(4,len(__import__('os').sched_getaffinity(0))))
    target=torch.tensor(y,dtype=torch.long);ce=[];acc=[]
    for logits in d['logits']:
        x=torch.tensor(logits,dtype=torch.float32)
        losses=[torch.nn.functional.cross_entropy(x[i:i+100],target[i:i+100]).item()*len(target[i:i+100]) for i in range(0,len(y),100)]
        ce.append(sum(losses)/len(y));acc.append(float((x.argmax(1)==target).float().mean())*100)
    mean=float(np.mean(ce))
    if not np.isfinite(mean):raise ValueError('nonfinite cross-entropy')
    return dict(final_loss_reduction_pct=100*(1-mean/2.20679),mean_final_ce=mean,ce_per_replica=ce,accuracy_pct=float(np.mean(acc)),guard='declared_protocol_pass; model probe and patch audit required')
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result=dict(final_loss_reduction_pct=float('inf'),guard='failed',reason=str(e))
    print(json.dumps(result))
