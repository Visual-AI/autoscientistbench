#!/usr/bin/env python3
"""Validate the tensor prediction schema and exact id set."""
import sys
import numpy as np

def check(pred, ids):
 required={'id','pred','seq_len','features'}
 if set(pred.files)!=required:raise ValueError('expected exactly id,pred,seq_len,features')
 pid=pred['id'];p=pred['pred']
 if pid.ndim!=1 or pid.dtype.kind not in 'US':raise ValueError('id must be a string vector')
 if len(set(pid.tolist()))!=len(pid):raise ValueError('duplicate ids')
 if set(pid.tolist())!=set(ids.tolist()):raise ValueError('id set mismatch')
 if p.shape!=(len(ids),96,7):raise ValueError('prediction shape must be (N,96,7)')
 if p.dtype.kind!='f' or not np.isfinite(p).all():raise ValueError('predictions must be finite floats')
 sl=pred['seq_len'];ft=pred['features']
 if sl.shape!=() or sl.dtype.kind not in 'iu' or int(sl)<1:raise ValueError('seq_len must be positive integer scalar')
 if ft.shape!=() or ft.dtype.kind not in 'US':raise ValueError('features must be string scalar')
 return pid,p

def main():
 try:
  with np.load(sys.argv[2],allow_pickle=False) as d:ids=d['id']
  with np.load(sys.argv[1],allow_pickle=False) as p:check(p,ids)
 except Exception as e:print(str(e),file=sys.stderr);return 1
 print('ok');return 0
if __name__=='__main__':sys.exit(main())
