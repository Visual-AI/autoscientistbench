#!/usr/bin/env python3
"""Accuracy of the validation-selected checkpoint; mean across seeds is external.
The manifest's invalid score is 0, exposed as manifest_score. The benchmark
uses Infinity as its failure sentinel so every protocol violation fails.
Training trace authenticity requires the model probe and patch audit.
"""
from pathlib import Path
import sys
import json
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import read_predictions,check_ids,PROTOCOL

def score(pred,labels):
    ids,p,meta=read_predictions(pred)
    with np.load(labels,allow_pickle=False) as z:
        check_ids(ids,z['id']);y=z['y']
        if y.shape!=p.shape or y.dtype.kind not in 'iu' or np.any((y<0)|(y>9)):raise ValueError('invalid labels')
        mean_steps=float(z['baseline_mean_total_local_steps'])
    if not math.isfinite(mean_steps) or mean_steps<=0:raise ValueError('invalid frozen baseline step limit')
    limit=1.2*mean_steps
    reasons=[]
    if meta['total_local_steps']>limit:reasons.append('local-work budget exceeded')
    if meta['rounds']!=50:reasons.append('requires 50 target rounds')
    if meta['clients']!=4:reasons.append('requires four clients')
    if meta['protocol']!=PROTOCOL:reasons.append('queue/partition protocol declaration changed')
    if not 1<=meta['selected_round']<=50:reasons.append('selected round outside 1..50')
    acc=100.0*int(np.count_nonzero(p==y))/len(y)
    return {'accuracy':float('inf') if reasons else acc,'manifest_score':0.0 if reasons else acc,
            'guard_pass':not reasons,'guard_reasons':reasons,'total_local_steps':meta['total_local_steps'],
            'baseline_local_steps':mean_steps,'step_limit':limit,'selected_round':meta['selected_round']}

if __name__=='__main__':
    try:r=score(*sys.argv[1:])
    except Exception as e:r={'accuracy':float('inf'),'manifest_score':0.0,'guard_pass':False,'guard_reasons':[str(e)]}
    print(json.dumps(r))
