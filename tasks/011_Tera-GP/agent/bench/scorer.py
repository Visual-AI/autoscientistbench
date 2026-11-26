#!/usr/bin/env python3
"""Sum of six mean raw-energy RMSEs per atom, subject to resource limits."""
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from formats import predictions

METRIC='energy_rmse_sum'

def score(pred_path,labels_path):
    pred,resource=predictions(pred_path)
    train_s,peak_gb,cpus=resource
    if peak_gb>56 or cpus>64 or cpus<1:
        return {METRIC:float('inf'),'guard':'peak_memory_or_cpu_budget',
                'train_s':train_s,'peak_gb':peak_gb,'cpus':cpus}
    groups={}; seen=set()
    with open(labels_path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','y','dataset','replica','n_atoms']:raise ValueError('Label header')
        for row in r:
            id_=row['id'];d=int(row['dataset']);rep=int(row['replica']);atoms=int(row['n_atoms'])
            if id_ in seen or id_ not in pred:raise ValueError('Duplicate or missing ID')
            if not 0<=d<6 or not 0<=rep<3 or atoms!=[56,60,87,118,148,370][d]:raise ValueError('Group metadata')
            y=float(row['y'])
            if not math.isfinite(y):raise ValueError('Non-finite label')
            seen.add(id_);groups.setdefault((d,rep),[]).append((pred[id_]-y)**2)
    if seen!=set(pred) or len(groups)!=18:raise ValueError('Incomplete or extra predictions/groups')
    rmse={f'd{d}_r{r}':math.sqrt(math.fsum(v)/len(v))/[56,60,87,118,148,370][d] for (d,r),v in groups.items()}
    means={f'd{d}':math.fsum(rmse[f'd{d}_r{r}'] for r in range(3))/3 for d in range(6)}
    return {METRIC:math.fsum(means.values()),'dataset_means':means,'replica_rmse':rmse,
            'train_s':train_s,'peak_gb':peak_gb,'cpus':cpus}

if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={METRIC:float('inf'),'guard':'invalid_prediction','reason':str(e)}
    print(json.dumps(result,allow_nan=True))
