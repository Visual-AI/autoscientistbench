"""Equal dataset / repetition mean NCIW, with per-dataset mean PICP >= .87 on test and >= .90 on val, and <= .99."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_predictions
DATASETS='computer ailerons airfoil ca_housing concrete elevator energy_efficiency insurance kin8nm miami_housing naval_propulsion parkinsons powerplant sulfur superconductor qsar allstate'.split()

def nciw(y,lower,upper,center):
    # Released compute_nciw, alpha=.05, but a target beyond a zero-width side is not covered at any scale, as the paper defines it.
    left=center-lower;right=upper-center
    candidates=[]
    for i in range(len(y)):
        if y[i]<center[i]:c=(center[i]-y[i])/left[i] if left[i]>0 else float('inf')
        elif y[i]>center[i]:c=(y[i]-center[i])/right[i] if right[i]>0 else float('inf')
        else:c=0
        candidates.append(c)
    factors=np.sort(candidates);idx=np.searchsorted(np.arange(1,len(y)+1)/len(y),.95)
    factor=factors[min(idx,len(y)-1)]
    width=np.mean(upper-lower);span=np.max(y)-np.min(y)
    return float(factor*(width/span if span else width))

PICP_FLOOR={'test':.87,'val':.90};PICP_CEILING=.99

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

def score(pred_path,labels_path):
    pred=read_predictions(pred_path);lab=np.load(labels_path,allow_pickle=False)
    expected={f'{i}_r{r}' for d in DATASETS for i in lab['id_'+d] for r in range(10)}
    if set(pred)!=expected:raise ValueError('id set mismatch')
    details={}
    for d in DATASETS:
        y=lab['y_'+d];ids=lab['id_'+d];widths=[];coverages=[]
        for r in range(10):
            yr=y[r] if y.ndim==2 else y
            p=np.array([pred[f'{i}_r{r}'] for i in ids]);center,lower,upper=p.T
            widths.append(nciw(yr,lower,upper,center));coverages.append(float(np.mean((yr>=lower)&(yr<=upper))))
        details[d]={'nciw':float(np.mean(widths)),'picp':float(np.mean(coverages)),'nciw_repetitions':widths,'picp_repetitions':coverages}
    floor=PICP_FLOOR[split_of(labels_path)];ok=all(floor<=v['picp']<=PICP_CEILING for v in details.values());value=float(np.mean([v['nciw'] for v in details.values()]))
    return {'nciw':value if ok and np.isfinite(value) else float('inf'),'nciw_unguarded':value,'coverage_guard':ok,'min_dataset_picp':min(v['picp'] for v in details.values()),'max_dataset_picp':max(v['picp'] for v in details.values()),'datasets':details}
if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result={'nciw':float('inf'),'coverage_guard':False,'error':str(e)}
    print(json.dumps(result))
