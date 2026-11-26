"""Fig. 4 power with literal fixed type-I and BH-FDR guards."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_predictions

def rejection_metrics(pvalues,nonnull):
    powers=[];type1=[];fdps=[]
    for p,b in zip(pvalues,nonnull):
        sig=p<0.05
        powers.append(float(np.mean(sig[b])))
        type1.append(float(np.mean(sig[~b])))
        # Same strict adjusted-p < .05 comparison as multipletests(fdr_bh).
        order=np.argsort(p);sorted_p=p[order]
        adjusted=np.minimum.accumulate((sorted_p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
        bh=np.empty(len(p),dtype=bool);bh[order]=adjusted<0.05
        fdps.append(float(np.sum(bh & ~b)/max(1,np.sum(bh))))
    return np.array(powers),np.array(type1),np.array(fdps)

def summarize(powers,type1,fdps):
    power=float(np.mean(powers));err=float(np.mean(type1));fdr=float(np.mean(fdps))
    ok=err<=0.069 and fdr<=0.085
    return dict(power=power if ok else 0.0,raw_power=power,type1_err=err,fdr_BH=fdr,guards_ok=ok,
                power_rep_std=float(np.std(powers,ddof=1)) if len(powers)>1 else 0.0,
                power_rep_se=float(np.std(powers,ddof=1)/np.sqrt(len(powers))) if len(powers)>1 else 0.0,
                type1_max=0.069,fdr_BH_max=0.085)

def score(pred,labels):
    with np.load(labels,allow_pickle=False) as z:
        ids=z['hypothesis_id'];nonnull=z['B']!=0
    values=read_predictions(pred,ids.ravel()).reshape(*ids.shape,3)
    p=values[:,:,2]
    return summarize(*rejection_metrics(p,nonnull))

def main(pred,labels):
    try:out=score(pred,labels)
    except Exception as e:out=dict(power=float('inf'),guards_ok=False,reason=str(e))
    print(json.dumps(out))

if __name__=='__main__':main(*sys.argv[1:])
