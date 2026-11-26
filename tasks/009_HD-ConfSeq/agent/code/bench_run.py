"""Protocol entry points for the released confidence-sequence implementation."""
import argparse,hashlib,json,os,time,contextlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from joblib import parallel_backend
from adaptive_sample.conf_sequences.ab_testing import _build_ab_hparams, _evaluate_ab_stopping
from adaptive_sample.conf_sequences.interval_factory import IntervalFactory
from adaptive_sample.conf_sequences.hedged_capital.hedged_factory import HedgedFactory
from adaptive_sample.conf_sequences.hedged_capital.hedged_capital_1d import ExactCachedHedgedCapitalCI

CONFIG=dict(method='HEDGE_nd_ELLIP_BBX_SAFE',alpha=.05,r_tol=.02,threshold=.5)

def one(task):
 iid,X,config,seed=task
 start=time.monotonic();D=X.shape[1]
 h=_build_ab_hparams(D,len(X),alpha=config['alpha'])
 # The pre-existing Monte Carlo seed setting is now driven by BENCH_SEED.
 h['volume_mc_seed']=int.from_bytes(hashlib.sha256(f'{seed}:{iid}'.encode()).digest()[:4],'little')
 h['volume_workers']=1
 h['interval_types']=[config['method']]
 np.random.seed(h['volume_mc_seed'])
 ExactCachedHedgedCapitalCI._get_log_wealth.cache_clear()
 method,_,_=IntervalFactory.create_instance(config['method'],h,X)
 calcs=HedgedFactory.build_and_cache_Khedgeds(h,X)
 ans={}
 if D==10:
  rows=[]
  # Same per-time evaluation as run(); discard work after the first decision.
  with parallel_backend('sequential'):
   for t in range(min(500,len(X))):
    low,up,extra=method._bounds_at_t(t,calcs,1/config['alpha'])
    vol,std=method.compute_volume(t,calcs,low,up,extra)
    rows.append((low,up,vol,extra['ellipse_params'][:,0],extra['radii'],h['parabola_power'],extra['proj_lowers'],extra['proj_uppers']))
    if np.any(low>config['threshold']) or np.any(up<config['threshold']) or vol**(1/D)<=config['r_tol']:break
  keys=['lower','upper','volume','center','radii','power','box_lower','box_upper']
  ans={k:np.asarray([r[j] for r in rows]) for j,k in enumerate(keys)}
  r=_evaluate_ab_stopping(ans['lower'],ans['upper'],ans['volume'],config['r_tol'],config['threshold'])
  ans.update(stop=np.asarray(r['stop_t']),decision=np.asarray({'significant':1,'no_significance':2,'no_decision':0}[r['decision']]),above=np.isin(np.arange(D),r['above']),below=np.isin(np.arange(D),r['below']),volume_seed=np.asarray(h['volume_mc_seed']))
 else:
  # is_member_at() inherits the exact hedged portfolio predicate. Export its
  # coefficients, never evaluate membership at a supplied or guessed truth.
  for c in calcs:c._advance_from_last_to(len(X)-1)
  ans=dict(coverage_type=np.asarray('hedged'),lam=np.stack([c._lams[:len(X)-1] for c in calcs],axis=1),c=np.array([c.c for c in calcs]),theta=np.array([c.theta for c in calcs]),weights=method.weights.copy())
 ExactCachedHedgedCapitalCI._get_log_wealth.cache_clear()
 return iid,ans,time.monotonic()-start

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict']);p.add_argument('--out');p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
 seed=int(os.environ['BENCH_SEED']);cpus=max(1,int(os.environ['BENCH_CPUS']))
 if a.stage=='train':
  out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
  (out/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
  (out/'training.json').write_text(json.dumps({'seed':seed,'cpus':cpus,'training':'none; configuration only'})+'\n')
  return
 config=json.loads((Path(a.model)/'config.json').read_text())
 with np.load(a.input,allow_pickle=False) as z:
  ids=z['id'].copy();tasks=[(str(i),z[str(i)].copy(),config,seed) for i in ids]
 result={'id':ids}
 with ProcessPoolExecutor(max_workers=min(cpus,len(tasks))) as pool:
  futures=[pool.submit(one,t) for t in tasks]
  for n,f in enumerate(as_completed(futures),1):
   iid,ans,seconds=f.result();result.update({iid+'__'+k:v for k,v in ans.items()})
   if 'stop' in ans:print(f'completed {n}/{len(tasks)} id={iid} stop={ans["stop"]} wall_s={seconds:.3f}',flush=True)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True)
 with open(a.output,'wb') as f:np.savez_compressed(f,**result)
if __name__=='__main__':main()
