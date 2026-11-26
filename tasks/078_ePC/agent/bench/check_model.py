#!/usr/bin/env python3
"""Reconstruct the fixed PC-research MLP; never import candidate Python."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_predictions
DIMENSIONS=[784,128,128,128,10]
PREFIXES=['layers.0.0','layers.1.0','layers.2.0','layers.3']

def fixed_weights(path):
    expected={prefix+'.'+kind: ((b,a) if kind=='weight' else (b,))
              for prefix,a,b in zip(PREFIXES,DIMENSIONS[:-1],DIMENSIONS[1:]) for kind in ('weight','bias')}
    with np.load(path,allow_pickle=False) as source:
        if set(source.files)!=set(expected):raise ValueError('fixed four-layer MLP tensor keys required')
        weights={key:source[key] for key in source.files}
    for key,shape in expected.items():
        value=weights[key]
        if value.shape!=shape or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('fixed MLP tensor shape/type/finiteness: '+key)
    if sum(value.size for value in weights.values())!=134794:raise ValueError('fixed parameter budget changed')
    return {key:torch.from_numpy(value) for key,value in weights.items()}

def classes_for(weights,x):
    result=[]
    with torch.no_grad():
        for start in range(0,len(x),64):
            value=torch.as_tensor(x[start:start+64]).float().div(255).sub_(.5).div_(.5).flatten(1)
            for i,prefix in enumerate(PREFIXES):
                value=torch.nn.functional.linear(value,weights[prefix+'.weight'],weights[prefix+'.bias'])
                if i<3:value=torch.nn.functional.gelu(value)
            result.append(value.argmax(1).numpy())
    return np.concatenate(result)

def check(model,inp,pred):
    root=Path(model).resolve();card=json.loads((root/'model_card.json').read_text())
    if card.get('format')!='epc-mlp-v1':raise ValueError('fixed PC-MLP certificate required')
    if card.get('dimensions')!=DIMENSIONS or card.get('activation')!='gelu' or card.get('parameters_per_model')!=134794:
        raise ValueError('fixed MLP structure/parameter budget changed')
    training=card.get('training',{});seed=training.get('bench_seed')
    if type(seed) is not int or seed<1:raise ValueError('positive integer bench_seed required')
    rule=str(training.get('rule','')).lower()
    if rule in ('backprop','backpropagation','bp') or (training.get('paradigm')!='predictive_coding' and rule not in
            ('error-based predictive coding','state-based predictive coding','predictive coding')):
        raise ValueError('predictive-coding training declaration required')
    if training.get('loss','CrossEntropyLoss') not in ('CrossEntropyLoss','cross_entropy','CE'):
        raise ValueError('fixed cross-entropy output objective changed')
    rows=card.get('replicas',[])
    if len(rows)!=5 or [row.get('slot') for row in rows]!=list(range(5)):raise ValueError('five ordered independent replicas required')
    ids,values=read_predictions(pred)
    with np.load(inp,allow_pickle=False) as source:
        if not np.array_equal(ids,source['id']):raise ValueError('input id mismatch')
        x=source['x']
    for i,row in enumerate(rows):
        if row.get('rng_seed')!=(5*(seed-1)+i)%2**32:raise ValueError('replica RNG seed mismatch')
        name=row.get('weights')
        if not isinstance(name,str) or not name or Path(name).is_absolute():raise ValueError('relative tensor path required')
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('artifact path escapes model directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row.get('sha256'):raise ValueError('artifact hash mismatch')
        if not np.array_equal(classes_for(fixed_weights(path),x),values[:,i]):raise ValueError('classes do not match fixed-network tensors')
    return {'pass':True,'members':5,'numerical_reconstruction':True,'training_provenance':'PC inference/local learning, CE and independence require patch audit'}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('--input',required=True);parser.add_argument('--pred',required=True);args=parser.parse_args()
    try:result=check(args.model,args.input,args.pred)
    except Exception as exc:result={'pass':False,'accuracy':float('inf'),'rule_score':0,'guard_failed':str(exc)}
    print(json.dumps(result));sys.exit(0 if result['pass'] else 1)
