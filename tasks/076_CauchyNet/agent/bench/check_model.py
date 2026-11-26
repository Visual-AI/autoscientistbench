#!/usr/bin/env python3
"""Trusted structural probe. Run candidate TorchScript in the judge sandbox.
check_model.py MODEL_DIR --input INPUT --pred PRED
Counts exported parameters and independently replays every exported model.
The export must represent every learned quantity; relabeling fitted tensors as
buffers or hard-coded constants is a patch-audit violation.
"""
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import predictions,read_csv

def check(model_dir,inp,pred,code=None):
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','8')))
    ids,p,reported=predictions(pred)
    rows=read_csv(inp,['id','x'])
    if ids!=[r['id'] for r in rows]:raise ValueError('input id mismatch')
    x=torch.tensor([[float(r['x'])] for r in rows],dtype=torch.float32)
    counts=[];diffs=[]
    code=Path(code or os.environ.get('BENCH_CODE_DIR','.'))
    if (code/'model_api.py').is_file():
        for seed in range(10):
            count=0
            with np.load(Path(model_dir)/f'model{seed}.parameters.npz',allow_pickle=False) as archive:
                for name in archive.files:
                    value=archive[name]
                    if not np.issubdtype(value.dtype,np.number) or not np.isfinite(value).all():raise ValueError('invalid learned parameter array')
                    count += value.size*(2 if np.iscomplexobj(value) else 1)
            if count>128 or count!=reported[seed]:raise ValueError('capacity budget or parameter manifest mismatch')
            counts.append(count)
        # This probe runs only in the isolated candidate execution environment.
        sys.path.insert(0,str(code))
        spec=importlib.util.spec_from_file_location('candidate_api',code/'model_api.py')
        api=importlib.util.module_from_spec(spec);sys.modules[spec.name]=api;spec.loader.exec_module(api)
        for indices in (np.arange(len(x)), np.arange(len(x)-1,-1,-1)[::2]):
            q=np.asarray(api.predict(Path(model_dir),x.numpy()[indices].copy()),dtype=np.float32)
            expected=p[indices]
            if q.shape!=expected.shape or not np.isfinite(q).all():raise ValueError('invalid generic prediction shape/values')
            if not np.allclose(q,expected,rtol=1e-5,atol=1e-6):raise ValueError('generic replay/subset mismatch')
            diffs.append(float(np.max(np.abs(q-expected))))
        if hasattr(api,'parameter_counts'):
            if not np.array_equal(np.asarray(api.parameter_counts(Path(model_dir))),counts):raise ValueError('false parameter declaration')
        return {'pass':True,'interface':'generic','real_parameters':counts,'max_abs_error':max(diffs)}
    for s in range(10):
        m=torch.jit.load(str(Path(model_dir)/f'model{s}.ts'),map_location='cpu');m.eval()
        count=sum(v.numel()*(2 if v.is_complex() else 1) for v in m.parameters())
        counts.append(count)
        if count>128:raise ValueError(f'model {s}: real parameter budget exceeded')
        if count!=reported[s]:raise ValueError(f'model {s}: false parameter declaration')
        with torch.no_grad():q=m(x)
        if isinstance(q,tuple):q=q[0]
        q=q.numpy().reshape(-1)
        if q.shape != p[:,s].shape or not np.isfinite(q).all():raise ValueError('invalid exported output')
        diffs.append(float(np.max(np.abs(q-p[:,s]))))
        if not np.allclose(q,p[:,s],rtol=1e-5,atol=1e-6):raise ValueError(f'model {s}: export/prediction mismatch')
    return {'pass':True,'real_parameters':counts,'max_abs_error':max(diffs)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('model_dir');ap.add_argument('--input',required=True);ap.add_argument('--pred',required=True);ap.add_argument('--code');a=ap.parse_args()
    try:r=check(a.model_dir,a.input,a.pred,a.code)
    except Exception as e:r={'pass':False,'mae':float('inf'),'reason':str(e)}
    print(json.dumps(r));return 0 if r['pass'] else 1
if __name__=='__main__':sys.exit(main())
