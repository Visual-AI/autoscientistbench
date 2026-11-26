"""EasyTSAD Table 2: dataset-macro and five-trial-mean event F1, delay gate."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import check
DATASETS=('AIOPS','TODS','WSD','UCR')
MARGIN={'AIOPS':5,'TODS':5,'WSD':3,'UCR':50}

def adjusted_labels(raw,n,post):
    labels=raw[-n:].copy();segments=[];flag=False;start=0
    for i in range(n):
        if i==n-1 and flag:segments.append((start,n))
        elif labels[i]==1 and not flag:flag=True;start=i
        elif labels[i]==0 and flag:flag=False;segments.append((start,i))
    if not segments:return labels
    labels[max(0,segments[0][0]):segments[0][0]+1]=1
    for j in range(1,len(segments)):
        labels[max(segments[j-1][1]+2,segments[j][0]):segments[j][0]+1]=1
    for j in range(len(segments)-1):
        labels[segments[j][1]-1:min(segments[j][1]+post,segments[j+1][0]-2)]=1
    labels[segments[-1][1]-1:min(segments[-1][1]+post,n)]=1
    return labels

def f1(scores,labels,delay=False):
    # Preserve official stable tie ordering, including its per-item threshold sweep.
    scores=scores-scores.min()
    positive=labels>.5;edges=np.flatnonzero(np.diff(np.r_[False,positive,False].astype(np.int8)))
    starts,ends=edges[::2],edges[1::2];normal=np.flatnonzero(~positive)
    ev=np.array([scores[a:min(b,a+5) if delay else b].max() for a,b in zip(starts,ends)])
    values=np.r_[scores[normal],ev];positions=np.r_[normal,starts]
    weights=np.r_[np.ones(len(normal)),ends-starts if delay else np.ones(len(starts))]
    positives=np.r_[np.zeros(len(normal)),ends-starts if delay else np.ones(len(starts))]
    order=np.argsort(positions,kind='stable');order=order[np.argsort(-values[order],kind='stable')]
    if not len(order):return 0.
    tp=np.cumsum(positives[order]);p=np.cumsum(weights[order]);total=positives.sum();eps=1e-15
    precision=tp/(p+eps);recall=tp/(total+eps)
    return float(np.max(2*precision*recall/(precision+recall+eps)))

F1D_FLOOR={'test':.727,'val':.59}

def split_of(labels_path):
    return 'val' if 'val' in Path(labels_path).parts[-3:-1] else 'test'

def score(pred,labels):
    ids,off,scores=check(pred,labels)
    with np.load(labels,allow_pickle=False) as z:
        lids=z['ids'];lo=z['offsets'];y=z['y'];ds=z['dataset']
    index={v:i for i,v in enumerate(ids.tolist())};values={d:[[],[]] for d in DATASETS};counts={d:0 for d in DATASETS}
    for j,ident in enumerate(lids):
        raw=y[lo[j]:lo[j+1]];d=str(ds[j])
        if np.all(raw<.5):continue  # Official skip occurs BEFORE 96-point alignment.
        i=index[ident];n=int(off[i+1]-off[i]);lab=adjusted_labels(raw,n,MARGIN[d]);counts[d]+=1
        values[d][0].append([f1(scores[t,off[i]:off[i+1]],lab) for t in range(5)])
        values[d][1].append([f1(scores[t,off[i]:off[i+1]],lab,True) for t in range(5)])
    # Empty positive populations have no F1; define their development contribution
    # as zero, report them explicitly, and retain the fixed four-dataset denominator.
    per={d:{'f1e':np.mean(values[d][0],axis=0).tolist() if counts[d] else [0.]*5,
            'f1d':np.mean(values[d][1],axis=0).tolist() if counts[d] else [0.]*5} for d in DATASETS}
    ev=np.mean([per[d]['f1e'] for d in DATASETS],axis=0);de=np.mean([per[d]['f1d'] for d in DATASETS],axis=0)
    e=float(ev.mean());d=float(de.mean());ok=d>=F1D_FLOOR[split_of(labels)]
    return dict(avg_f1e=e if ok else 0.,avg_f1e_unguarded=e,avg_f1d=d,guard_f1d_ok=ok,
                trial_f1e=ev.tolist(),trial_f1d=de.tolist(),per_dataset=per,evaluated_curves=counts,
                empty_datasets=[d for d in DATASETS if not counts[d]])

if __name__=='__main__':
    try:result=score(sys.argv[1],sys.argv[2])
    except Exception as e:result={'avg_f1e':float('inf'),'format_ok':False,'reason':str(e)}
    print(json.dumps(result))
