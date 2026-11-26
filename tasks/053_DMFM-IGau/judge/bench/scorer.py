#!/usr/bin/env python3
import json,sys
from protocol import load,rows,dimensions,fossil_ari
try:
    meta=load(sys.argv[2]);ids,iters,burn=dimensions(meta)
    truth={x['id']:x['labels'] for x in meta['networks']}
    if any(len(y)!=50 or y.count(1)!=25 or y.count(2)!=25 for y in truth.values()): raise ValueError('invalid truth')
    sums={i:0.0 for i in ids}
    for i,c,k in rows(sys.argv[1],meta): sums[i]+=fossil_ari(c,truth[i],k)
    per_network={i:sums[i]/(iters-burn) for i in ids}
    print(json.dumps({'dARI':sum(per_network.values())/len(ids),'valid':True,'networks':len(ids),'draws_per_network':iters-burn}))
except Exception as e:
    # The literal fossil implementation is nonnegative; -1 is the worst sentinel.
    print(json.dumps({'dARI':-1.0,'valid':False,'error':str(e)}))
