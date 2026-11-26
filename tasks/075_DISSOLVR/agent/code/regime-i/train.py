"""Released five-model CatBoost experiment with materialized training input."""
import argparse,json,multiprocessing as mp,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
BASE_SEEDS=[42,101,123,456,789]

def initialize(groups,threads):
 os.sched_setaffinity(0,groups.get())
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
  os.environ[key]=str(threads)

def fit(args):
 index,seed,out,threads=args
 import joblib
 from catboost import CatBoostRegressor
 X,y=joblib.load(out/'training_features.joblib')
 model=CatBoostRegressor(iterations=10000,learning_rate=0.02,depth=8,l2_leaf_reg=5,
                        verbose=False,random_state=seed,allow_writing_files=False,thread_count=threads)
 model.fit(X,y)
 joblib.dump(model,out/f'model_{index}.joblib')
 print(f'finished model {index}, seed {seed}, trees {model.tree_count_}',flush=True)
 return {'seed':seed,'tree_count':model.tree_count_}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
 budget=int(os.environ.get('BENCH_CPUS','16'));seed=int(os.environ.get('BENCH_SEED','1'))
 cpus=sorted(os.sched_getaffinity(0))[:budget];os.sched_setaffinity(0,cpus)
 threads=min(4,len(cpus));workers=min(5,len(cpus)//threads)
 # Featurization has no random choices. Limit BLAS as in the reproduced runner.
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
  os.environ[key]=str(threads)
 import joblib,pandas as pd
 from rdkit import RDLogger
 from featurizer import MoleculeFeaturizer
 RDLogger.DisableLog('rdApp.*')
 train=pd.read_csv(Path(os.environ['BENCH_DATA_DIR'])/'train/train.csv')
 X=MoleculeFeaturizer().transform(train.SMILES)
 joblib.dump((X,train.y),out/'training_features.joblib')
 seeds=[(s+seed-1)%(2**32) for s in BASE_SEEDS]
 ctx=mp.get_context('spawn');groups=ctx.Queue()
 for i in range(workers):groups.put(cpus[i*threads:(i+1)*threads])
 with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=initialize,initargs=(groups,threads)) as pool:
  records=list(pool.map(fit,[(i,s,out,threads) for i,s in enumerate(seeds)]))
 (out/'panel.json').write_text(json.dumps({'bench_seed':seed,'models':records,'features':list(X.columns)}))
 (out/'label_usage.json').write_text(json.dumps({'label_sources':['train/train.csv'],'row_ids':train.id.tolist()}))
 # Features are a transient training artifact, not needed for inference.
 (out/'training_features.joblib').unlink()
if __name__=='__main__':main()
