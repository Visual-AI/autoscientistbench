#!/usr/bin/env python3
"""Subset-consistency probe. Run with the reference Python after prediction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import validate

def digest_tree(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('model');ap.add_argument('--input',required=True);ap.add_argument('--pred',required=True);ap.add_argument('--code',required=True)
    args=ap.parse_args();model=Path(args.model).resolve();code=Path(args.code).resolve()
    baseline,subjects=validate(args.pred,args.input)
    before=digest_tree(model)
    max_error=0.
    with tempfile.TemporaryDirectory(prefix='fmri-consistency-') as tmp:
        inputs=Path(tmp)/'input';inputs.mkdir()
        selections={}
        for s in subjects:
            with np.load(Path(args.input)/f'subject_{s}.npz') as d:
                # Noncontiguous, label-independent subset. Keep ids sorted.
                idx=np.arange(0,len(d['id']),max(1,len(d['id'])//31))
                selections[s]=idx
                np.savez(inputs/f'subject_{s}.npz',**{k:d[k][idx] for k in d.files})
        pred=Path(tmp)/'pred.out'
        subprocess.run(['bash',str(code/'predict.sh'),'--model',str(model),'--input',str(inputs),'--output',str(pred)],check=True)
        candidate,_=validate(pred,inputs)
        for s in subjects:
            want=baseline[f'embedding_{s}'][selections[s]];got=candidate[f'embedding_{s}']
            assert want.shape==got.shape
            error=np.max(np.abs(want-got)/np.maximum(1.,np.abs(want)))
            max_error=max(max_error,float(error))
            assert np.allclose(want,got,rtol=1e-4,atol=1e-4),f'subject {s}: batch-dependent embeddings'
    assert digest_tree(model)==before,'predict rewrote model artifacts'
    print(json.dumps(dict(passed=True,max_relative_error=max_error)))

if __name__=='__main__':main()
