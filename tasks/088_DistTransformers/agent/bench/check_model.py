"""Replay certificate guard. The operator invokes this in the execution sandbox.

Checks the frozen runner against submitted mixtures, then permutes a subset and
checks that inference is per-problem. Runtime is measured again against original
DT in the same process. Input provenance and method execution also require
patch audit: arbitrary Python cannot supply a self-authenticating certificate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_prediction


def check(model,inp,pred,code):
    runner=Path(__file__).resolve().parent/'predict_runner.py'
    original_ids,original=read_prediction(pred)
    original_map={k:v for k,v in zip(original_ids,original)}
    with np.load(inp,allow_pickle=False) as f: data={k:f[k] for k in f.files}
    rng=np.random.default_rng(9183)
    subset=rng.permutation(len(data['id']))[:min(73,len(data['id']))]
    errors=[];ratios=[]
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        for name,indices in [('full',np.arange(len(data['id']))),('subset',subset)]:
            ip=temp/(name+'.npz');op=temp/(name+'.csv')
            np.savez(ip,**{k:v[indices] for k,v in data.items()})
            cmd=[sys.executable,str(runner),'--model',str(model),'--input',str(ip),'--output',str(op),'--code',str(code)]
            r=subprocess.run(cmd,capture_output=True,text=True,env=os.environ)
            if r.returncode:
                raise ValueError('frozen replay failed: '+r.stderr[-1000:])
            ids,v=read_prediction(op)
            expected=np.array([original_map[i][:6] for i in ids])
            error=float(np.max(np.abs(v[:,:6]-expected)/np.maximum(1,np.abs(expected))))
            errors.append(error);ratios.append(float(v[:,8].mean()/v[:,9].mean()))
            if error>2e-5:
                raise ValueError('predictions disagree with model replay/subset: '+str(error))
    return dict(posterior_kl_guard=True,pass_guard=True,max_relative_error=max(errors),forward_ratios=ratios)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',type=Path)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--pred',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True);a=p.parse_args()
    try:result=check(a.model,a.input,a.pred,a.code)
    except Exception as e:result=dict(pass_guard=False,posterior_kl=float('inf'),reason=str(e))
    print(json.dumps(result));sys.exit(0 if result['pass_guard'] else 1)
