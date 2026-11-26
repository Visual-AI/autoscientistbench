"""Literal budgeted first threshold crossing, with objective/accounting integrity checks."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_predictions
from oracle import METHODS
from generator import centered

def score(pred,labels):
    with np.load(labels,allow_pickle=False) as z:lab={k:z[k] for k in z.files}
    rows=read_predictions(pred,lab['id']);per=[]
    for i,iid in enumerate(lab['id']):
        C=centered(lab['A'][i],lab['b'][i]);b=lab['b'][i];fs=float(lab['fstar'][i])
        for method in METHODS:
            row=rows[(str(iid),method)];h=row['history'];g=v=0
            if not np.array_equal(h[0]['x'],np.ones(500)):raise ValueError('initial point must be ones')
            vals=[]
            for rec in h:
                actual=float(logsumexp(C.dot(np.asarray(rec['x']))-b))
                if abs(actual-rec['func'])>1e-10:raise ValueError('forged objective')
                g+=rec['events'].count('grad');v+=rec['events'].count('inverse')
                if (g,v)!=(rec['grad_calls'],rec['matrix_inverses']):raise ValueError('forged counters')
                vals.append(actual-fs)
            hit=next((k for k,x in enumerate(vals) if x<=1e-7),None)
            selected=h[hit if hit is not None else -1]
            iterations=hit if hit is not None else 500
            if row['iterations']!=iterations:raise ValueError('wrong first crossing')
            if abs(row['final_objective']-h[-1]['func'])>1e-10:raise ValueError('wrong final objective')
            if (row['grad_calls'],row['matrix_inverses'])!=(selected['grad_calls'],selected['matrix_inverses']):raise ValueError('wrong crossing counters')
            ok=hit is not None and selected['matrix_inverses']<=24 and selected['grad_calls']<=25
            per.append({'id':str(iid),'method':method,'iterations':iterations,'score':iterations if ok else 500,
                        'budget_ok':ok,'grad_calls':selected['grad_calls'],'matrix_inverses':selected['matrix_inverses'],
                        'final_residual':vals[-1]})
    main=[r['score'] for r in per if r['method']=='exact_func']
    diagnostics={m:float(np.mean([r['iterations'] for r in per if r['method']==m])) for m in METHODS}
    return {'iterations':float(np.mean(main)),'integrity_ok':True,'crossings':diagnostics,'per_instance':per}

if __name__=='__main__':
    try:
        with threadpool_limits(limits=1):result=score(*sys.argv[1:])
    except Exception as e:result={'iterations':float('inf'),'integrity_ok':False,'reason':str(e)}
    print(json.dumps(result))
