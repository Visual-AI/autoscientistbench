"""Frozen reader for the epoch prediction table and its trailing evidence record."""
import csv,json
import numpy as np

def read(path,ids):
    ids=[str(i) for i in ids]
    if len(set(ids))!=len(ids):raise ValueError('duplicate input ids')
    pred=np.empty((50,len(ids)),dtype=np.int64)
    with open(path,newline='') as f:
        if f.readline().strip()!='id,epoch,pred':raise ValueError('wrong header')
        for e in range(1,51):
            for j,i in enumerate(ids):
                r=next(csv.reader([f.readline()]))
                if len(r)!=3 or r[0]!=i or int(r[1])!=e:raise ValueError('wrong id/epoch/order, missing or duplicate row')
                v=int(r[2])
                if not 0<=v<10:raise ValueError('class must be integer 0..9')
                pred[e-1,j]=v
        tail=f.readline()
        if not tail.startswith('#certificate='):raise ValueError('missing certificate')
        cert=json.loads(tail[len('#certificate='):])
        if f.read().strip():raise ValueError('extra rows')
    return pred,cert

def accuracy_curve(pred,y,batch_size=128):
    # Deliberately retain the released unweighted mean of batch accuracies.
    return np.array([np.mean([np.mean(p[k:k+batch_size]==y[k:k+batch_size],dtype=np.float32) for k in range(0,len(y),batch_size)],dtype=np.float32) for p in pred])

def crossing(curve,interpolate=True):
    for j,a in enumerate(curve):
        if a>=0.5:
            if j==0 or not interpolate:return float(j+1)
            return float(j+(0.5-float(curve[j-1]))/(float(a)-float(curve[j-1])))
    return 51.0
