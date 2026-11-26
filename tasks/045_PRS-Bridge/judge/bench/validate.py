#!/usr/bin/env python3
"""Strict CSV prediction contract; usable independently of candidate code."""
import csv,sys
import numpy as np
HEADER=['id','pred_0','pred_1']
def read_prediction(path,expected):
 with open(path,newline='') as f:
  reader=csv.reader(f)
  if next(reader,None)!=HEADER: raise ValueError('header must be id,pred_0,pred_1')
  rows=list(reader)
 if any(len(r)!=3 for r in rows): raise ValueError('wrong column count')
 ids=[r[0] for r in rows]
 if len(ids)!=len(set(ids)): raise ValueError('duplicate id')
 if ids!=list(expected): raise ValueError('ids must match input exactly, in input order')
 p=np.array([[float(v) for v in r[1:]] for r in rows],dtype=np.float64)
 if p.shape!=(len(expected),2): raise ValueError('wrong prediction shape')
 if not np.isfinite(p).all(): raise ValueError('non-finite prediction')
 return p
def main():
 try:
  with np.load(sys.argv[2],allow_pickle=False) as x: expected=x['id']
  if expected.ndim!=1 or len(set(expected.tolist()))!=len(expected): raise ValueError('bad input ids')
  read_prediction(sys.argv[1],expected)
 except Exception as e:
  print('INVALID: '+str(e),file=sys.stderr); return 1
 print('VALID'); return 0
if __name__=='__main__': sys.exit(main())
