#!/usr/bin/env python3
"""Frozen overlap accuracy on the supplied reads."""
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import load_prediction


def score(pred,labels):
    labels=Path(labels)
    cfg=json.loads((labels/'labels.json').read_text())
    meta,rows=load_prediction(pred)
    if cfg['protocol']=='full':
        if cfg['n_reads']!=1000000:raise ValueError('full protocol requires all reads')
        if meta['threads']>12:raise ValueError('full protocol permits at most 12 threads')
    elif cfg['protocol']!='smoke':raise ValueError('unknown protocol')
    correct=0;seen=set()
    with gzip.open(labels/'intervals.tsv.gz','rt') as f:
        for line in f:
            key,ref,a,b=line.rstrip('\n').split('\t')
            if key in seen or key not in rows:raise ValueError('label/record mismatch')
            seen.add(key);start,end=int(a),int(b)
            p_ref,p_start,p_end=rows[key]
            # Match the authors' overlap(): touching endpoints count as overlap.
            if ref!='*' and ref==p_ref and max(start,p_start)<=min(end,p_end):correct+=1
    if seen!=rows.keys() or len(seen)!=cfg['n_reads']:raise ValueError('incomplete read coverage')
    accuracy=100.0*correct/len(seen)
    runtime=meta['mapping_time_s_tool_reported']
    passed=True
    return {'score':accuracy if passed else 0.0,'mapping_accuracy_pct':accuracy,
            'mapping_time_s_tool_reported':runtime,
            'constraints_ok':passed,'reads':len(seen),'correct':correct,'protocol':cfg['protocol'],
            'runtime_provenance':'candidate tool report; timer and execution authenticity require patch audit'}

if __name__=='__main__':
    try:print(json.dumps(score(*sys.argv[1:]),allow_nan=False))
    except Exception as exc:print(json.dumps({'score':0.0,'constraints_ok':False,'invalid':str(exc)}))
