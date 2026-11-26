"""CLEAR standard variant (a): fixed public partitions and portable saved estimators.

The bootstrap work queue follows the reproduced execution protocol. Selection,
bootstrap fits, aggregation, residual augmentation and CLEAR calibration retain
the released hyperparameters. Independent jobs share the CPU budget.
"""
import argparse,copy,csv,json,multiprocessing as mp,os,pickle,sys,tempfile,time
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import numpy as np
from sklearn.utils import resample
from quantile_forest import RandomForestQuantileRegressor
from xgboost import XGBRegressor
from pygam import ExpectileGAM
from PCS.regression import PCS_UQ
from clear.clear import CLEAR
from utils import optimize_gam_smoothing
from training_data import load_training_data

DATASETS='computer ailerons airfoil ca_housing concrete elevator energy_efficiency insurance kin8nm miami_housing naval_propulsion parkinsons powerplant sulfur superconductor qsar allstate'.split()
DATA={};OUT=None;SCRATCH=None;BASE=None

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('wb') as f:pickle.dump(obj,f,protocol=5)
def read(p):
    with Path(p).open('rb') as f:return pickle.load(f)
def pool(fn,tasks,n,title):
    start=time.monotonic()
    with mp.get_context('fork').Pool(n) as p:
        for k,result in enumerate(p.imap_unordered(fn,tasks,chunksize=1),1):
            if k%max(1,len(tasks)//20)==0 or k==len(tasks):print(f'{title}: {k}/{len(tasks)} elapsed={time.monotonic()-start:.1f}s',flush=True)
def select(key):
    state=DATA[key];xt,yt,xv,yv=[state[k] for k in ('xt','yt','xv','yv')]
    with (OUT/'logs'/f'{key}.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        # The authors' 300 s search limit: allstate and qsar take hours, so ExpectileGAM is not a candidate there.
        lam,converged=optimize_gam_smoothing(xt,yt,n_splines=10,timeout=300)
        models={'QRF':RandomForestQuantileRegressor(random_state=BASE,n_estimators=100,default_quantiles=.5,min_samples_leaf=10),
                'QXGB':XGBRegressor(random_state=BASE,objective='reg:quantileerror',n_estimators=100,tree_method='hist',quantile_alpha=.5,min_child_weight=10)}
        if converged:models['ExpectileGAM']=ExpectileGAM(expectile=.5,n_splines=10,lam=lam,spline_order=3,basis='ps')
        pcs=PCS_UQ(models=models,num_bootstraps=100,alpha=.05,seed=state['seed'],top_k=1,calibration_method='multiplicative')
        pcs._train(xt,yt);pcs._pred_check(xv,yv);pcs.top_k_models=pcs._get_top_k()
        name=next(iter(pcs.top_k_models))
        if name=='QRF':kind='rf';params={'n_estimators':100,'random_state':BASE,'min_samples_leaf':10}
        elif name=='QXGB':kind='xgb';params={'n_estimators':100,'tree_method':'hist','random_state':BASE,'min_child_weight':10}
        else:kind='qgam';params={'n_splines':10,'lam':lam,'spline_order':3,'basis':'ps'}
        state.update(template=next(iter(pcs.top_k_models.values())),kind=kind,params=params,top_model=name,lam=lam,gam_converged=converged)
        dump(OUT/'state'/f'{key}.pkl',state)
    return key

def ep_fit(job):
    key,i=job;s=DATA[key];xb,yb=resample(s['xt'],s['yt'],random_state=s['seed']+i)
    m=copy.deepcopy(s['template']);m.fit(xb,yb)
    pred=m.predict(np.concatenate([s['xt'],s['xv']]))
    np.load(SCRATCH/f'{key}_ep.npy',mmap_mode='r+')[i]=pred
    dump(OUT/'estimators'/key/f'ep_{i}.pkl',m)

def ep_finish(key):
    s=DATA[key];a=np.load(SCRATCH/f'{key}_ep.npy',mmap_mode='r')
    v=np.column_stack([np.quantile(a,q,axis=0) for q in [.025,.5,.975]])
    n=len(s['yt']);s['ep_train']=v[:n];s['ep_val']=v[n:]
    pcs=PCS_UQ(models={},alpha=.05);pcs.calibrate(s['ep_val'],s['yv']);s['pcs_gamma']=pcs.gamma
    dump(OUT/'state'/f'{key}.pkl',s)

def ale_pred(models,xx,ep):
    xx=np.column_stack([xx,ep])
    if type(models[0]) is RandomForestQuantileRegressor and models[0].estimators_ is models[1].estimators_:
        pred=models[0].predict(xx,quantiles=[(1-.95)/2,.5,1-(1-.95)/2])
    else:pred=np.column_stack([m.predict(xx) for m in models])
    pred=pred.astype(np.float64,copy=False)+ep[:,None]
    pred[:,0]=np.minimum(pred[:,0],pred[:,1]);pred[:,2]=np.maximum(pred[:,2],pred[:,1])
    return pred

def ale_fit(job):
    key,i=job;s=DATA[key]
    import gam_predict_only
    gam_predict_only.install()
    m=CLEAR(desired_coverage=.95,n_bootstraps=100,random_state=BASE,n_jobs=1);m.fit_on_residuals=True
    ep=s['ep_train'][:,1]
    models=m._fit_single_bootstrap(s['xt'],s['yt']-ep,s['kind'],s['params'],i,ep)
    pred=ale_pred(models,s['xv'],s['ep_val'][:,1])
    np.load(SCRATCH/f'{key}_ale.npy',mmap_mode='r+')[i]=pred
    dump(OUT/'estimators'/key/f'ale_{i}.pkl',models)

def calibrate(key):
    s=DATA[key];av=np.median(np.load(SCRATCH/f'{key}_ale.npy',mmap_mode='r'),axis=0);pv=s['ep_val']
    with (OUT/'logs'/f'{key}.log').open('a') as log,redirect_stdout(log),redirect_stderr(log):
        m=CLEAR(desired_coverage=.95,n_bootstraps=100,random_state=BASE,n_jobs=1,lambdas=np.concatenate((np.linspace(0,.09,10),np.logspace(-1,2,4001))))
        m.calibrate(s['yv'].flatten(),median_epistemic=pv[:,1],aleatoric_median=av[:,1],aleatoric_lower=av[:,0],aleatoric_upper=av[:,2],epistemic_lower=pv[:,0],epistemic_upper=pv[:,2])
    dump(OUT/'calibration'/f'{key}.pkl',m)
    s['ale_val']=av;dump(OUT/'state'/f'{key}.pkl',s)

def ep_predict(job):
    key,i=job;pred=read(OUT/'estimators'/key/f'ep_{i}.pkl').predict(DATA[key]['x'])
    np.load(SCRATCH/f'{key}_ep.npy',mmap_mode='r+')[i]=pred

def ale_predict(job):
    key,i=job;s=DATA[key];models=read(OUT/'estimators'/key/f'ale_{i}.pkl')
    np.load(SCRATCH/f'{key}_ale.npy',mmap_mode='r+')[i]=ale_pred(models,s['x'],s['ep'][:,1])

def main():
    global DATA,OUT,SCRATCH,BASE
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','predict']);ap.add_argument('--out');ap.add_argument('--model');ap.add_argument('--input');ap.add_argument('--output');ap.add_argument('--datasets',default=','.join(DATASETS));a=ap.parse_args()
    cpus=max(1,int(os.environ['BENCH_CPUS']));BASE=777+1000*(int(os.environ['BENCH_SEED'])-1)
    ds=a.datasets.split(',');keys=[f'{d}_{r}' for d in ds for r in range(10)]
    OUT=Path(a.out if a.stage=='train' else a.model).resolve();OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='clear-stage-') as tmp:
        SCRATCH=Path(tmp)
        if a.stage=='train':
            for d in ['logs','state','estimators','calibration']:(OUT/d).mkdir(exist_ok=True)
            # Selection and calibration fit reusable state. Reserve their rows
            # within training; the external validation split is never loaded.
            training,split_counts=load_training_data(Path(os.environ['BENCH_DATA_DIR']),ds)
            for d in ds:
                for r in range(10):DATA[f'{d}_{r}']=dict(training[d],seed=BASE+r)
            keys.sort(key=lambda k:DATA[k]['xt'].size,reverse=True)
            pool(select,keys,min(32,cpus),'selection')
            DATA={k:read(OUT/'state'/f'{k}.pkl') for k in keys}
            for k,s in DATA.items():np.lib.format.open_memmap(SCRATCH/f'{k}_ep.npy',mode='w+',dtype='float64',shape=(100,len(s['yt'])+len(s['yv'])))
            jobs=[(k,i) for k in keys for i in range(100)]
            pool(ep_fit,jobs,cpus,'PCS fits');pool(ep_finish,keys,min(16,cpus),'PCS aggregation')
            DATA={k:read(OUT/'state'/f'{k}.pkl') for k in keys}
            for k,s in DATA.items():np.lib.format.open_memmap(SCRATCH/f'{k}_ale.npy',mode='w+',dtype='float64',shape=(100,len(s['yv']),3))
            pool(ale_fit,jobs,cpus,'CLEAR residual fits');pool(calibrate,keys,min(32,cpus),'CLEAR calibration')
            (OUT/'manifest.json').write_text(json.dumps({'datasets':ds,'seed':int(os.environ['BENCH_SEED']),'repetitions':10,'bootstraps':100,'training_source':'train','internal_calibration_split_seed':20261102,'internal_split_counts':split_counts}))
        else:
            manifest=json.loads((OUT/'manifest.json').read_text());inp=np.load(a.input,allow_pickle=False)
            assert manifest['seed']==int(os.environ['BENCH_SEED'])
            ds=[d for d in manifest['datasets'] if 'x_'+d in inp.files];keys=[f'{d}_{r}' for d in ds for r in range(10)]
            DATA={f'{d}_{r}':{'x':inp['x_'+d]} for d in ds for r in range(10)}
            for k,s in DATA.items():
                np.lib.format.open_memmap(SCRATCH/f'{k}_ep.npy',mode='w+',dtype='float64',shape=(100,len(s['x'])))
                np.lib.format.open_memmap(SCRATCH/f'{k}_ale.npy',mode='w+',dtype='float64',shape=(100,len(s['x']),3))
            jobs=[(k,i) for k in keys for i in range(100)];pool(ep_predict,jobs,cpus,'PCS predictions')
            for k,s in DATA.items():s['ep']=np.column_stack([np.quantile(np.load(SCRATCH/f'{k}_ep.npy',mmap_mode='r'),q,axis=0) for q in [.025,.5,.975]])
            pool(ale_predict,jobs,cpus,'CLEAR predictions')
            with open(a.output,'w',newline='') as f:
                w=csv.writer(f);w.writerow(['id','pred','lower','upper'])
                for d in ds:
                    for r in range(10):
                        k=f'{d}_{r}';s=DATA[k];ep=s['ep'];av=np.median(np.load(SCRATCH/f'{k}_ale.npy',mmap_mode='r'),axis=0);m=read(OUT/'calibration'/f'{k}.pkl')
                        lower,upper=m.predict(s['x'],external_epistemic={'median':ep[:,1],'lower':ep[:,0],'upper':ep[:,2]},external_aleatoric={'median':av[:,1],'lower':av[:,0],'upper':av[:,2]})
                        for id_,p,l,u in zip(inp['id_'+d],ep[:,1],lower,upper):w.writerow([f'{id_}_r{r}',repr(float(p)),repr(float(l)),repr(float(u))])
if __name__=='__main__':main()
