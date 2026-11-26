#!/usr/bin/env python3
"""Optional model-dir/certificate consistency probe; scorer already enforces the numeric guards."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from certificate import *
def main():
 a=argparse.ArgumentParser();a.add_argument('model');a.add_argument('--input',required=True);a.add_argument('--pred',required=True);a.add_argument('--labels');args=a.parse_args()
 try:
  configure(); root=Path(args.model); card=json.loads((root/'model_card.json').read_text())
  assert card['format']=='fc12-mask/v1' and card['certificate']=='certificate.npz'
  model=load(root/'certificate.npz'); pred=load(args.pred); x=load(args.input)
  validate_arrays(model);validate_arrays(pred,x['id'])
  for k in MODEL_KEYS: assert np.array_equal(model[k],pred[k]),'prediction/model certificate mismatch'
  assert card['dense_sha256']==dense_hash(model) and card['seed']==int(model['seed'])
  lab=args.labels or str(Path(os.environ.get('BENCH_DATA_DIR',Path(__file__).resolve().parent.parent/'data'))/'val/labels.npz')
  authenticate(pred,load(lab));mask_check(pred);verified_logits(pred,x['x'])
  print(json.dumps({'ok':True,'dense_sha256':dense_hash(model)}));return 0
 except Exception as e: print(json.dumps({'ok':False,'error':str(e)}));return 1
if __name__=='__main__':sys.exit(main())
