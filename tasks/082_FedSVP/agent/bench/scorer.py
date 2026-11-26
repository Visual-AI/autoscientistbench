"""GenAcc with the PerAcc >= 86.5 constraint; structural rules need a probe/audit."""
import sys,csv,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from format import read_predictions

def score(pred,labels):
    ids,p=read_predictions(pred)
    with open(labels,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','y','client']:raise ValueError('labels header')
        rows=list(reader)
    truth=np.array([[int(r[k]) for k in ['id','y','client']] for r in rows],dtype=np.int64)
    if truth.shape!=(len(ids),3) or not np.array_equal(ids,truth[:,0]):raise ValueError('label IDs mismatch')
    y=truth[:,1];client=truth[:,2]
    if np.any(y<0) or np.any(y>=101) or set(client.tolist())!=set(range(10)):raise ValueError('label domain')
    gen=100*int((p[:,0]==y).sum())/len(y)
    acc=[100*int((p[client==c,1+c]==y[client==c]).sum())/int((client==c).sum()) for c in range(10)]
    per=sum(acc)/10
    ok=per>=86.5
    return {'genacc':gen if ok else float('inf'),'raw_genacc':gen,'peracc':per,'peracc_threshold':86.5,'guard_pass':ok,'worst_score':0 if not ok else None,'personal_by_client':acc}
if __name__=='__main__':
    try:r=score(*sys.argv[1:])
    except Exception as e:r={'genacc':float('inf'),'guard_pass':False,'worst_score':0,'error':str(e)}
    print(json.dumps(r))
