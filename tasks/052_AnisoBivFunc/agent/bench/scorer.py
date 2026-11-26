#!/usr/bin/env python3
"""Mean across 60 cells of median absolute angle error (200 replications/cell)."""
import json,math,statistics,sys
from collections import defaultdict
from contract import ID,predictions
METRIC='mean_median_angle_risk'
WORST=1e100
def score(pred_path,labels_path):
    lab=json.load(open(labels_path))
    protocol=lab['protocol']
    if protocol=='figure3-full-v1':
        cells,reps=60,200
    elif protocol=='smoke-v1':
        cells,reps=2,1
    else:raise ValueError('unrecognized frozen-label protocol')
    if lab['cells']!=cells or lab['reps_per_cell']!=reps or len(lab['records'])!=cells*reps:raise ValueError('incomplete replication design')
    truths={};groups=defaultdict(list)
    allowed=[math.pi/30,math.pi/5,math.pi/4,math.pi/3,math.pi/2-math.pi/30]
    for row in lab['records']:
        if set(row)!={'id','cell','alpha'} or not ID.fullmatch(row['id']) or row['id'] in truths:raise ValueError('invalid label row or duplicate ID')
        alpha=row['alpha']
        if isinstance(alpha,bool) or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not any(abs(alpha-a)<1e-14 for a in allowed):raise ValueError('invalid truth angle')
        truths[row['id']]=alpha;groups[row['cell']].append(row['id'])
    if len(groups)!=cells or any(len(v)!=reps for v in groups.values()):raise ValueError('incomplete cells')
    if protocol=='figure3-full-v1' and set(groups)!={str(i) for i in range(1,61)}:raise ValueError('invalid cell set')
    pred=predictions(pred_path)
    if set(pred)!=set(truths):raise ValueError('prediction/label ID mismatch')
    return statistics.mean(statistics.median(abs(pred[i]-truths[i]) for i in ids) for ids in groups.values())
if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as exc:
        print('invalid: '+str(exc),file=sys.stderr);result=WORST
    print(json.dumps({METRIC:result},allow_nan=False))
