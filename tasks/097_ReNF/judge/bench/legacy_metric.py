#!/usr/bin/env python3
"""Standardized horizon-96 multivariate MSE; protocol failures yield Infinity.
Lookback declarations are also verified by an operator truncation probe and patch audit.
"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prediction_validate import check

def main(pred_path,labels_path):
 result={'mse':float('inf'),'guard_ok':False}
 try:
  with np.load(labels_path,allow_pickle=False) as lab, np.load(pred_path,allow_pickle=False) as pred:
   ids=lab['id'];pid,p=check(pred,ids)
   if int(pred['seq_len'])>720:raise ValueError('seq_len exceeds 720')
   if str(pred['features'])!='M':raise ValueError('features must be M')
   lookup={k:i for i,k in enumerate(pid.tolist())};p=p[[lookup[k] for k in ids.tolist()]]
   mean,std=lab['train_mean'],lab['train_std']
   y=(lab['y'].astype(np.float64)-mean)/std
   p=(p.astype(np.float64)-mean)/std
   err=p-y;mse=float(np.mean(err**2));mae=float(np.mean(np.abs(err)))
   if not np.isfinite(mse):raise ValueError('nonfinite metric')
   result.update(mse=mse,mae=mae,guard_ok=True)
 except Exception as e:result['reason']=str(e)
 print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
