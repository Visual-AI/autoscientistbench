"""Protocol wrapper around the unchanged chromosome-wise PRSBridge program."""
import argparse,concurrent.futures as cf,json,os,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parent
SETTINGS=[('0.125','0.8'),('0.25','0.8')]
def fit(data,out,chrom,alpha,percent):
 dest=out/f'chr{chrom}'/f'percent{percent}'/f'alpha{alpha}'; dest.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
 cmd=[sys.executable,str(CODE/'PRSBridge.py'),'--percent',percent,'--chr',str(chrom),'--alpha',alpha,'--ref',str(data/'train/ref/chr'),'--sumdat',str(data/f'train/sumdat/chr{chrom}/hm3_sumdat.txt'),'--h2','0','--h2_se','0','--method','cg','--output',str(dest)]
 start=time.monotonic()
 with (dest/'fit.log').open('w') as f: subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 return {'chrom':chrom,'alpha':alpha,'percent':percent,'seconds':time.monotonic()-start}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args()
 data=Path(os.environ['BENCH_DATA_DIR']).resolve(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
 cpus=max(1,int(os.environ['BENCH_CPUS'])); seed=int(os.environ['BENCH_SEED']); times=[]
 with cf.ThreadPoolExecutor(max_workers=cpus) as pool:
  tasks=[pool.submit(fit,data,out,c,al,pc) for c in range(1,23) for al,pc in SETTINGS]
  for f in cf.as_completed(tasks):
   r=f.result(); times.append(r); print(json.dumps(r),flush=True)
 (out/'model.json').write_text(json.dumps({'seed':seed,'settings':SETTINGS,'fits':times},indent=2))
if __name__=='__main__': main()
