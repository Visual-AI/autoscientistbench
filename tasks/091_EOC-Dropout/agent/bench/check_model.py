"""Fixed-network tensor reconstruction; candidate Python is never imported."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from schema import CONFIG,load,guard

DIMENSIONS=[3072]+[256]*6+[10]

def fixed_weights(path):
    weights=torch.load(path,map_location='cpu',weights_only=True)
    expected={f'layers.{i}.{kind}': ((b,a) if kind=='weight' else (b,))
              for i,(a,b) in enumerate(zip(DIMENSIONS[:-1],DIMENSIONS[1:])) for kind in ('weight','bias')}
    if not isinstance(weights,dict) or set(weights)!=set(expected):raise ValueError('fixed MLP weight keys required')
    for key,shape in expected.items():
        value=weights[key]
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('fixed MLP tensor shape/type/finiteness: '+key)
    return weights

def logits_for(weights,x):
    parts=[]
    with torch.no_grad():
        for start in range(0,len(x),100):
            value=torch.as_tensor(x[start:start+100],dtype=torch.float32).reshape(-1,3072)
            for layer in range(7):
                value=torch.nn.functional.linear(value,weights[f'layers.{layer}.weight'],weights[f'layers.{layer}.bias'])
                if layer<6:value=torch.relu(value)
            parts.append(value.numpy())
    return np.concatenate(parts)

def check(model_dir,inp,pred):
    d=load(pred);guard(d);root=Path(model_dir).resolve()
    card=json.loads((root/'model_card.json').read_text())
    if card.get('format')!='eoc-mlp-v1' or card.get('config')!=CONFIG:raise ValueError('fixed critical-MLP certificate required')
    if card.get('bench_seed')!=int(d['bench_seed']) or card.get('seeds')!=d['seeds'].tolist():raise ValueError('card/output seed mismatch')
    if not np.array_equal(np.asarray(card.get('rates')),d['rates']):raise ValueError('card/output allocation mismatch')
    rows=card.get('models',[])
    if len(rows)!=25:raise ValueError('need25 independent model entries')
    with np.load(inp,allow_pickle=False) as z:
        if not np.array_equal(z['ids'],d['ids']):raise ValueError('input id mismatch')
        x=z['x']
    for i,row in enumerate(rows):
        if row.get('seed')!=int(d['seeds'][i]) or row.get('epoch')!=75:raise ValueError('member seed/final-epoch mismatch')
        name=row.get('file')
        if not isinstance(name,str) or not name or Path(name).is_absolute():raise ValueError('relative weights path required')
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('weight path escapes model directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row.get('sha256'):raise ValueError('artifact hash mismatch')
        actual=logits_for(fixed_weights(path),x)
        if not np.allclose(actual,d['logits'][i],rtol=1e-5,atol=1e-5):raise ValueError('predictions do not match fixed-network weights')
    return dict(passed=True,members=25,numerical_reconstruction=True,training_provenance='initialization, dropout and training controls require patch audit')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model_dir');p.add_argument('--input',required=True);p.add_argument('--pred',required=True);a=p.parse_args()
    try:out=check(a.model_dir,a.input,a.pred)
    except Exception as e:print(json.dumps(dict(passed=False,final_loss_reduction_pct=float('inf'),reason=str(e))));sys.exit(1)
    print(json.dumps(out))
