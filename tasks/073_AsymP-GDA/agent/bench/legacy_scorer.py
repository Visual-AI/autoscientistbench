"""Last-iterate combined NashConv, recomputed by trusted LiteEFG.

Reported counter consistency, budget and oracle flag are scorer guards.
Counter authenticity and exclusive interface access require replay + patch audit.
"""
import contextlib,json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from legacy_validate import read_prediction
with contextlib.redirect_stdout(sys.stderr):
    from private_leduc import evaluate,MAX_UPDATES


def score(pred_path,labels_path):
    try:
        with np.load(labels_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        policies,counters=read_prediction(pred_path,data)
        ids=[str(x) for x in data['id']]
        for iid,(updates,traversals,oracle) in counters.items():
            if updates>MAX_UPDATES or updates!=traversals/2 or oracle!=1:
                return {'nashconv':float('inf'),'guards_ok':False,'reason':'budget, traversal consistency or oracle violation'}
        scores=[float(evaluate(str(data['game'][n]),policies[i])) for n,i in enumerate(ids)]
        if not all(np.isfinite(scores)):raise ValueError('nonfinite NashConv')
        return {'nashconv':float(np.mean(scores)),'guards_ok':True,'per_instance':scores,
                'updates':[counters[i][0] for i in ids],'traversals':[counters[i][1] for i in ids]}
    except Exception as e:
        return {'nashconv':float('inf'),'guards_ok':False,'reason':str(e)}


def main(pred,labels):
    # LiteEFG may write C++ diagnostics to stdout. The external contract is one JSON line.
    sys.stdout.flush();saved=os.dup(1)
    try:
        with open(os.devnull,'w') as sink:
            os.dup2(sink.fileno(),1)
            result=score(pred,labels)
    finally:os.dup2(saved,1);os.close(saved)
    print(json.dumps(result))

if __name__=='__main__':main(*sys.argv[1:])
