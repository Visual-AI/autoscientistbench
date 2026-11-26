"""Load the five saved models and predict supplied molecules without labels."""
import argparse,json,os
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 threads=min(4,int(os.environ.get('BENCH_CPUS','16')))
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:threads])
 for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]=str(threads)
 import joblib,pandas as pd
 from rdkit import RDLogger
 from featurizer import MoleculeFeaturizer
 RDLogger.DisableLog('rdApp.*')
 inp=pd.read_csv(a.input);X=MoleculeFeaturizer().transform(inp.SMILES)
 panel=json.loads((a.model/'panel.json').read_text());assert list(X.columns)==panel['features']
 meta=json.loads((a.model/'train_meta.json').read_text());out=inp[['id']].copy()
 for j in range(5):
  model=joblib.load(a.model/f'model_{j}.joblib');out[f'pred_{j}']=model.predict(X,thread_count=threads)
 out['train_s']=meta['train_s'];out['train_only']=meta['train_only']
 a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False,float_format='%.17g')
if __name__=='__main__':main()
