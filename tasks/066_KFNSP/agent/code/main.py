"""Strict train-only fit for independent validation and test, using official SVR-FKD."""
import argparse,csv,json,os,random,time
from pathlib import Path
import numpy as np
import joblib
from models.decomposition_models import RegularizedSVR

PARAMS=dict(alpha_prime=0.05,gamma=0.05,C=0.75,eps=0.01,nystrom_comp=None)

def read(path,cols=None):
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        cols=cols or [c for c in r.fieldnames if c not in ('id','y','p','fold')]
        rows=list(r)
    X=np.asarray([[float(r[c]) for c in cols] for r in rows],dtype=np.float32)
    return rows,cols,X

def fit_one(args):
    data,out,group,seed=args
    random.seed(seed);np.random.seed(seed)
    rows,cols,X=read(data)
    keep=np.array([group=='all' or r['fold']!=group for r in rows])
    y=np.asarray([float(r['y']) for r in rows],dtype=np.float32)
    p=np.asarray([float(r['p']) for r in rows],dtype=np.float32)
    t=time.monotonic()
    m=RegularizedSVR(**PARAMS)
    m.train(X[keep],y[keep],p[keep],iterations=30)
    assert m.model.fit_status_==0
    # Intermediate transforms are unused at inference; same serialization as reproduction.
    m.regularizer.transformations=[]
    joblib.dump(m,Path(out)/f'{group}.joblib',compress=0)
    return {'group':group,'training_ids':[r['id'] for r,k in zip(rows,keep) if k],'train_s':time.monotonic()-t}

def train(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    data=Path(a.data).resolve()/'train/train.csv'
    rows,cols,_=read(data)
    seed=int(os.environ.get('BENCH_SEED',a.seed))
    records=[fit_one((str(data),str(out),'all',seed))]
    card={'format':'training-only-refit/v1','training_source':'train','feature_names':cols,'models':records,'params':PARAMS,'iterations':30,'seed':seed,'workers':1,'threads_per_worker':1}
    (out/'model_card.json').write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps({'trained':len(records),'workers':1}),flush=True)

def predict(a):
    root=Path(a.model);card=json.loads((root/'model_card.json').read_text())
    rows,_,X=read(a.input,card['feature_names'])
    model=joblib.load(root/'all.joblib')
    pred=model.predict(X).flatten().astype(np.float32)
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pred']);w.writerows((r['id'],repr(float(p))) for r,p in zip(rows,pred))

if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train');t.add_argument('--data',required=True);t.add_argument('--seed',type=int,default=1);t.add_argument('--out',required=True)
    p=sp.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args();train(a) if a.cmd=='train' else predict(a)
