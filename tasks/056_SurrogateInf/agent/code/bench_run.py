"""Protocol adapter for the authors' per-instance PII estimator."""
import argparse,csv,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.decomposition import PCA
from PII import PII, fit_rf

CONFIG=dict(r=10,link='logit',crossfit=1)

def infer(job):
    X,Y,C,hids,config,seed=job
    np.random.seed(seed)
    t0=time.monotonic()
    U=lambda Y,Y_test: PCA(n_components=config['r']).fit_transform(Y_test)
    res=PII(Y,X,C,U,fit_rf,fit_rf,Y_C=Y[:,C],link=config['link'],crossfit=config['crossfit'])
    rows=list(zip(hids,res['hbeta'][0],res['var_beta'][0],res['p_values'][0]))
    print(json.dumps(dict(instance=str(hids[0]).rsplit('_',1)[0],elapsed_s=time.monotonic()-t0)),flush=True)
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','predict'])
    for key in ['out','model','input','output']:ap.add_argument('--'+key)
    a=ap.parse_args();seed=int(os.environ.get('BENCH_SEED','1'));cpus=max(1,int(os.environ.get('BENCH_CPUS','1')))
    if a.mode=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps(dict(**CONFIG,seed=seed)))
        return
    config=json.loads((Path(a.model)/'config.json').read_text())
    with np.load(a.input,allow_pickle=False) as z:
        jobs=[(z['X'][i],z['Y'][i],z['C'][i],z['hypothesis_id'][i],config,seed) for i in range(len(z['id']))]
    with ProcessPoolExecutor(max_workers=min(cpus,len(jobs))) as pool:
        result=list(pool.map(infer,jobs))
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','beta','var','p_value'])
        for rows in result:w.writerows(rows)

if __name__=='__main__':main()
