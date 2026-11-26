#!/usr/bin/env python3
"""Mean of five separate RMSEs, subject to reported protocol guards.
Elapsed-time authenticity needs the external timing probe; label provenance needs
patch audit. A prediction file cannot prove either claim by itself.
"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from table import read,prediction

def score(p,y):
 rows=prediction(p);lab=read(y,['id','y'])
 if [r['id'] for r in rows]!=[r['id'] for r in lab]:raise ValueError('prediction/label ids differ')
 seconds=float(rows[0]['train_s']);only=float(rows[0]['train_only'])
 if only!=1:raise ValueError('training used labels outside train.csv')
 truth=np.array([float(r['y']) for r in lab],dtype=np.float64)
 values=[float(np.sqrt(np.mean((np.array([float(r[f'pred_{j}']) for r in rows])-truth)**2))) for j in range(5)]
 result=float(np.mean(values))
 if not np.isfinite(result):raise ValueError('nonfinite RMSE')
 return {'rmse':result,'rmse_per_model':values,'train_s':seconds,'train_only':True}
if __name__=='__main__':
 try:result=score(*sys.argv[1:])
 except Exception as e:result={'rmse':float('inf'),'guard_failure':str(e)}
 print(json.dumps(result))
