"""Protocol adapter. Numerical filters, MPC, and accounting are upstream code."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import src.backtest as bt
from src.filter.single_ckf import calibrate_sigma_raw
from src.filter.cokf import CrossAssetCollaborativeKF, estimate_rho_raw
from src.mpc import PaperMPC

PANELS=('NASDAQ','CSI300','TPX100','Global30')
LAMBDAS=(.05,.1,.3,.6)
CS=(1.,1.5,2.)
ORDERS=(1,2,3)
L=7

class PaperMPCFloored(PaperMPC):
 def __init__(self,*args,sigma_l_mult=1.5,**kwargs):
  super().__init__(*args,**kwargs)
  self.sigma_l_mult=sigma_l_mult
 def solve(self,u_current,price_pred,sigma_l_paper,capital):
  return super().solve(u_current,price_pred,sigma_l_paper*self.sigma_l_mult,capital)

def read_input(path):
 panels={}
 with open(path,newline='') as f:
  for row in csv.DictReader(f):panels.setdefault(row['panel'],[]).append(row)
 result={}
 for panel,rows in panels.items():
  steps=[int(r['step']) for r in rows]
  assert steps==list(range(len(rows))), 'History must be chronological and complete'
  raw=np.array([[float(r[f'p{i:02d}']) for i in range(30)] for r in rows])
  active=np.array([int(r['active']) for r in rows],dtype=bool)
  start=int(np.flatnonzero(active)[0]) if active.any() else len(rows)
  assert np.array_equal(np.flatnonzero(active),np.arange(start,len(rows)-1)) if active.any() else True
  result[panel]=(raw,start,[r['id'] for r in rows])
 return result

def calibrate(job):
 name,raw=job
 sigma=np.array([calibrate_sigma_raw(raw[:,j],order=1) for j in range(raw.shape[1])])
 return name,dict(sigma_v=sigma[:,0].tolist(),sigma_w=sigma[:,1].tolist(),rho=estimate_rho_raw(raw,len(raw)).tolist())

def run_cell(job):
 name,raw,start,params,lam,c=job
 sigma_v=np.array(params['sigma_v']);sigma_w=np.array(params['sigma_w']);rho=np.array(params['rho'])
 cokf=CrossAssetCollaborativeKF(M=raw.shape[1],orders=ORDERS,sigma_v_per_order=tuple(sigma_v.copy() for _ in ORDERS),sigma_w=sigma_w,rho=rho,horizon=L)
 # Recording-only hook: return exactly the upstream clipping result.
 orders=np.zeros_like(raw)
 original=bt._apply_no_leverage_clip
 def record(u_committed,cur_N,price,V_k):
  clipped=original(u_committed,cur_N,price,V_k)
  k=sys._getframe(1).f_locals['k']
  orders[k]=clipped
  return clipped
 bt._apply_no_leverage_clip=record
 try:
  result=bt.backtest_cokf_raw(cokf=cokf,mpc_factory=lambda:PaperMPCFloored(horizon=L,lam=lam,b=.5,sigma_l_mult=c),raw_panel=raw,test_start_idx=start,V_0_per_asset=1000.)
 finally:bt._apply_no_leverage_clip=original
 metric=bt.metric_row(result['V_portfolio'][start:])
 return dict(panel=name,lam=lam,c=c,sharpe=metric['sharpe'],mdd=metric['max_dd'],orders=orders)

def pool_map(fn,jobs):
 with ProcessPoolExecutor(max_workers=min(int(os.environ.get('BENCH_CPUS','1')),len(jobs))) as pool:return list(pool.map(fn,jobs))

def train(a):
 data=Path(os.environ['BENCH_DATA_DIR'])
 cal=read_input(data/'train/input.csv');val=read_input(data/'val/input.csv')
 params=dict(pool_map(calibrate,[(p,cal[p][0]) for p in PANELS]))
 print('Calibration complete',flush=True)
 jobs=[(p,val[p][0],val[p][1],params[p],lam,c) for c in CS for p in PANELS for lam in LAMBDAS]
 cells=pool_map(run_cell,jobs)
 # Original c ablation aggregates across panels and lambda cells. Selection
 # now uses validation only; Sharpe primary, MDD tie-break, then grid order.
 cstats=[]
 for c in CS:
  group=[r for r in cells if r['c']==c]
  cstats.append((float(np.mean([r['sharpe'] for r in group])),float(np.mean([r['mdd'] for r in group])),c))
 best_c=sorted(cstats,key=lambda r:(-r[0] if np.isfinite(r[0]) else np.inf,r[1],CS.index(r[2])))[0][2]
 for p in PANELS:
  group=[r for r in cells if r['panel']==p and r['c']==best_c]
  best=sorted(group,key=lambda r:(-r['sharpe'] if np.isfinite(r['sharpe']) else np.inf,r['mdd'],LAMBDAS.index(r['lam'])))[0]
  params[p].update(lam=best['lam'],c=best_c)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'model.json').write_text(json.dumps(dict(panels=params,seed=int(os.environ.get('BENCH_SEED','1'))),indent=2))
 (out/'selection.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='orders'} for r in cells],indent=2))
 print(json.dumps({p:{k:params[p][k] for k in ['lam','c']} for p in PANELS}),flush=True)

def predict(a):
 model=json.loads((Path(a.model)/'model.json').read_text())['panels'];inp=read_input(a.input)
 cells=pool_map(run_cell,[(p,x[0],x[1],model[p],model[p]['lam'],model[p]['c']) for p,x in inp.items()])
 with open(a.output,'w',newline='') as f:
  w=csv.writer(f);w.writerow(['id']+[f'u{i:02d}' for i in range(30)])
  for r in cells:
   _,start,ids=inp[r['panel']]
   for k in range(start,len(ids)-1):w.writerow([ids[k],*[repr(float(v)) for v in r['orders'][k]]])
 print('Orders written',flush=True)

if __name__=='__main__':
 seed=int(os.environ.get('BENCH_SEED','1'));random.seed(seed);np.random.seed(seed)
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
 tr=sp.add_parser('train');tr.add_argument('--out',required=True)
 pr=sp.add_parser('predict');pr.add_argument('--model',required=True);pr.add_argument('--input',required=True);pr.add_argument('--output',required=True)
 a=ap.parse_args();train(a) if a.command=='train' else predict(a)
