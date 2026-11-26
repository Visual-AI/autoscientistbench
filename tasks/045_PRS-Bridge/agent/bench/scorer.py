#!/usr/bin/env python3
"""Mean validation R² after tuning-half selection over 100 frozen splits."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from validate import read_prediction

def evaluate(p,y,masks):
 if p.shape!=(len(y),2) or masks.shape!=(100,len(y)) or masks.dtype!=bool: raise ValueError('invalid shapes')
 if not np.isfinite(p).all() or not np.isfinite(y).all(): raise ValueError('non-finite values')
 if np.any(masks.sum(axis=1)!=round(len(y)/2)): raise ValueError('splits must be 50/50')
 tune=np.zeros((100,2)); valid=np.zeros_like(tune)
 for s,mask in enumerate(masks):
  yt,yv=y[mask],y[~mask]
  for j in range(2):
   pt,pv=p[mask,j],p[~mask,j]
   if min(np.std(yt),np.std(yv),np.std(pt),np.std(pv))==0: raise ValueError('undefined correlation for constant column')
   tune[s,j]=np.corrcoef(yt,pt)[0,1]**2
   valid[s,j]=np.corrcoef(yv,pv)[0,1]**2
 if not np.isfinite(tune).all() or not np.isfinite(valid).all(): raise ValueError('undefined correlation')
 best=np.argmax(tune,axis=1); selected=valid[np.arange(100),best]
 return {'mean_r2':float(selected.mean()),'split_sd':float(selected.std(ddof=1)),
         'selected_counts':np.bincount(best,minlength=2).tolist(),
         'per_candidate_mean_r2':valid.mean(axis=0).tolist(),'guards_ok':True}

def main():
 try:
  with np.load(sys.argv[2],allow_pickle=False) as lab:
   ids,y,masks=lab['id'],lab['y'],lab['tune']
  p=read_prediction(sys.argv[1],ids)
  result=evaluate(p,y,masks)
 except Exception as e: result={'mean_r2':float('inf'),'guards_ok':False,'reason':str(e)}
 print(json.dumps(result))
if __name__=='__main__': main()
