"""Frozen output and AP helpers, independent of candidate code."""
import json
import numpy as np

def load_prediction(path, shapes):
    p=np.load(path,allow_pickle=False)
    if len(p.files)!=len(set(p.files)):raise ValueError('duplicate ZIP keys')
    if set(p.files)!=set(shapes)|{'__meta__'}:raise ValueError('instance id set mismatch')
    meta=json.loads(str(p['__meta__']))
    wall=meta.get('wall_s');cpus=meta.get('cpus')
    if isinstance(wall,bool) or not isinstance(wall,(int,float)) or not np.isfinite(wall) or wall<0:raise ValueError('invalid wall time receipt')
    if isinstance(cpus,bool) or not isinstance(cpus,int) or cpus<1 or cpus>16:raise ValueError('invalid CPU receipt')
    for sid,shape in shapes.items():
        a=p[sid]
        if a.shape!=tuple(shape) or a.dtype!=np.float32:raise ValueError('expected float32 similarity matrix '+sid)
        if not np.isfinite(a).all():raise ValueError('nonfinite similarities '+sid)
    return p,meta

def average_precision(scores,relevant):
    # torchmetrics 1.4.0.post0 retrieval_average_precision, full-gallery path.
    # Preserve topk tie-breaking and float32 arithmetic of the authors' metric.
    import torch
    target=relevant[scores.topk(scores.shape[-1],sorted=True,dim=-1)[1]]
    if not target.sum():return torch.tensor(0.0,device=scores.device)
    positions=torch.arange(1,len(target)+1,device=target.device,dtype=torch.float32)[target>0]
    return torch.div(torch.arange(len(positions),device=positions.device,dtype=torch.float32)+1,positions).mean()

# AP arithmetic above is adapted from TorchMetrics 1.4.0.post0.
# Copyright The Lightning team, Apache License 2.0:
# https://www.apache.org/licenses/LICENSE-2.0
