#!/usr/bin/env python3
"""Equal-weight mean dataset mAP (%). Timing is diagnostic.

Output shape/id/finite guards give Infinity. Runtime receipt needs the independent
judge timing probe; model/data provenance need patch audit. No threshold on a
particular dataset was specified, so per-dataset mAP is diagnostic only.
"""
import sys,os,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_prediction,average_precision

def score(pred,labels,wall_override=None):
    started=time.monotonic()
    import torch
    torch.set_num_threads(max(1,min(16,int(os.environ.get('OMP_NUM_THREADS','1')))))
    with np.load(labels,allow_pickle=False) as lab:
        ids=sorted(k[:-7] for k in lab.files if k.endswith('__query'))
        shapes={sid:(len(lab[sid+'__query']),len(lab[sid+'__gallery'])) for sid in ids}
        p,meta=load_prediction(pred,shapes)
        maps={}
        for sid in ids:
            sim=torch.from_numpy(p[sid]);ql=torch.from_numpy(lab[sid+'__query']);gl=torch.from_numpy(lab[sid+'__gallery']);selfs=lab[sid+'__self']
            aps=[]
            for i in range(len(ql)):
                s=sim[i];r=ql[i]==gl;j=int(selfs[i])
                if j>=0:s=torch.cat((s[:j],s[j+1:]));r=torch.cat((r[:j],r[j+1:]))
                aps.append(average_precision(s,r).item())
            maps[sid]=float(np.mean(aps)*100)
        wall=float(meta['wall_s']+time.monotonic()-started if wall_override is None else wall_override)
        raw=float(np.mean(list(maps.values())))
        return {'retrieval_map':raw,'raw_map':raw,'per_instance_map':maps,'eval_wall_s':wall,'timing_source':'prediction_receipt' if wall_override is None else 'judge_probe'}

if __name__=='__main__':
    try:result=score(*sys.argv[1:])
    except Exception as e:result={'retrieval_map':float('inf'),'invalid':str(e)}
    print(json.dumps(result))
