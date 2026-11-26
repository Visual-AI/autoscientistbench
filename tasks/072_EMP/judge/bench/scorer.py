#!/usr/bin/env python3
"""Feasibility-gated mask sparsity. raw_score=0 on failure; Infinity signals FAIL to the harness."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from certificate import load,configure,validate_arrays,authenticate,mask_check,forward,verified_logits,ce,TOTAL

def score(pred_path,labels_path):
 try:
  configure(); p=load(pred_path); labels=load(labels_path)
  validate_arrays(p,labels['id']); authenticate(p,labels)
  kept=mask_check(p)
  pruned=verified_logits(p,labels['x']); dense=forward(p,labels['x'],'dense::')
  dense_loss=ce(dense,labels['y']); pruned_loss=ce(pruned,labels['y'])
  epsilon=abs(pruned_loss-dense_loss); sparsity=100*(1-kept/TOTAL)
  result={'sparsity':sparsity,'raw_score':sparsity,'epsilon':epsilon,'dense_loss':dense_loss,'pruned_loss':pruned_loss,'kept':kept,'total':TOTAL,'mask_only':True,'fixed_dense':True,'certificate_verified':True}
  if epsilon>0.1: result.update(sparsity=float('inf'),raw_score=0.,guard_failure='epsilon exceeds 0.1')
  return result
 except Exception as e: return {'sparsity':float('inf'),'raw_score':0.,'guard_failure':str(e)}
if __name__=='__main__': print(json.dumps(score(sys.argv[1],sys.argv[2])))
