"""Protocol entry points for the authors' transductive IRT + log-time calibration."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats
import torch
import fit_irt


def read(path, prefix):
    with Path(path).open(newline='') as f:
        return [(prefix + r['id'], json.loads(r['responses'])) for r in csv.DictReader(f)]


def fit(rows, labels_path, work, seed):
    """Reconstruct the original observation order; call the unmodified authors' fit."""
    subjects = {}
    for item, cells in rows:
        for subject, block, rank, response in cells:
            subjects.setdefault(subject, []).append((block, rank, item, response))
    path = work / 'responses.jsonl'
    with path.open('w') as f:
        for subject in sorted(subjects):
            cells = sorted(subjects[subject])
            assert len({(b,r) for b,r,_,_ in cells}) == len(cells)
            f.write(json.dumps({'subject_id': str(subject), 'responses': {t:v for _,_,t,v in cells}})+'\n')
    fit_irt.PARAMS_DIR = work
    old_argv = sys.argv
    sys.argv = ['fit_irt.py', '--input_path', str(path), '--seed', str(seed), '--epochs', '1000', '--device', 'cpu']
    try:
        fit_irt.main()
    finally:
        sys.argv = old_argv
    params = pd.read_csv(work/'responses.csv', index_col=0)
    with Path(labels_path).open(newline='') as f:
        labels = list(csv.DictReader(f))
    x = np.array([params.loc['m'+r['id'],'b'] for r in labels])
    y = np.log(np.array([float(r['y']) for r in labels]))
    reg = stats.linregress(x,y)
    return params, {'slope':float(reg.slope),'intercept':float(reg.intercept)}


def train(a):
    data=Path(a.data)/'train';out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name in ['input.csv','labels.csv','auxiliary.csv']:
        shutil.copyfile(data/name,out/name)
    rows=read(out/'input.csv','m')+read(out/'auxiliary.csv','a')
    params,reg=fit(rows,out/'labels.csv',out,a.seed)
    (out/'model.json').write_text(json.dumps({'seed':a.seed,'calibration':reg})+'\n')
    print('Saved training context and fitted calibration',flush=True)


def predict(a):
    model=Path(a.model)
    seed=int(os.environ.get('BENCH_SEED',json.loads((model/'model.json').read_text())['seed']))
    # A query task is unlabeled. Its identity is local to this batch and cannot join training labels.
    query=read(a.input,'q')
    rows=read(model/'input.csv','m')+read(model/'auxiliary.csv','a')+query
    with tempfile.TemporaryDirectory(prefix='bridge_predict_') as tmp:
        params,reg=fit(rows,model/'labels.csv',Path(tmp),seed)
        pred=np.exp(reg['intercept']+reg['slope']*np.array([params.loc[i,'b'] for i,_ in query]))
    with Path(a.output).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pred'])
        for (i,_),p in zip(query,pred):w.writerow([i[1:],repr(float(p))])


if __name__=='__main__':
    torch.set_num_threads(int(os.environ.get('BENCH_CPUS','8')))
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train');t.add_argument('--out',required=True);t.add_argument('--data',required=True);t.add_argument('--seed',type=int,required=True)
    q=sub.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True)
    args=p.parse_args();train(args) if args.cmd=='train' else predict(args)
