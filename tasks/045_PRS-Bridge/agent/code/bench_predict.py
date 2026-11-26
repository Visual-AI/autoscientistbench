"""Reproduction's PRS accumulation, with materialized inputs and CSV output."""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
CODE2DOS=np.array([2,255,1,0],dtype=np.uint8)
def decode(seg,n):
 codes=(seg[:,:,None] >> np.array([0,2,4,6],dtype=np.uint8)) & 3
 return CODE2DOS[codes].reshape(seg.shape[0],-1)[:,:n].astype(np.float32)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
 model=Path(a.model); data=Path(os.environ['BENCH_DATA_DIR'])
 settings=json.loads((model/'model.json').read_text())['settings']
 if len(settings)!=2: raise ValueError('Expected two candidate PRS columns')
 with np.load(a.input,allow_pickle=False) as inp:
  iids=inp['id']; n=len(iids); total=None
  for chrom in range(1,23):
   with np.load(data/f'train/variants{chrom}.npz',allow_pickle=False) as v: ids,a1,a2=v['snp'],v['a1'],v['a2']
   pos={s:i for i,s in enumerate(ids)}; k=len(settings)
   W=np.zeros((len(ids),k),dtype=np.float32); off=np.zeros(k); nused=np.zeros(k,dtype=np.int64)
   for s,(al,p) in enumerate(settings):
    cf=model/f'chr{chrom}'/f'percent{p}'/f'alpha{al}'/'coef.txt'
    with cf.open() as f:
     next(f)
     for line in f:
      q=line.split()
      if len(q)<4: continue
      j=pos.get(q[0])
      if j is None: continue
      try: w=float(q[3])
      except ValueError: continue
      # Preserve the reproduction's q[1] allele comparison, despite its docstring.
      if q[1]==a1[j]: W[j,s]=w
      elif q[1]==a2[j]: W[j,s]=-w; off[s]+=2.0*w
      else: continue
      nused[s]+=1
   seg=inp[f'chr{chrom}']; assert seg.shape==(len(ids),(n+3)//4)
   acc=np.zeros((n,k))
   for st in range(0,len(ids),4000):
    en=min(len(ids),st+4000); wk=W[st:en]; nz=np.abs(wk).sum(axis=1)>0
    if not nz.any(): continue
    dos=decode(seg[st:en],n); dos[dos==255]=0.0
    acc+=dos[nz].T@wk[nz]
   acc+=off[None,:]; total=acc if total is None else total+acc
   print(f'chr{chrom}: used={nused.tolist()}',flush=True)
  with open(a.output,'w',newline='') as f:
   writer=csv.writer(f); writer.writerow(['id','pred_0','pred_1'])
   for iid,row in zip(iids,total): writer.writerow([iid,*[f'{v:.8g}' for v in row]])
if __name__=='__main__': main()
