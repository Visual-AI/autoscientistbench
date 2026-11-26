"""Independent horizon scheduling; the four chains share the CPU budget."""
import argparse, concurrent.futures, json, os, subprocess, sys
from pathlib import Path

HS=(96,192,336,720)
CODE=Path(__file__).resolve().parent

def command(h,model,out):
    args=[sys.executable,'-u',str(CODE/'run.py'),'--is_training','1',
          '--root_path',os.environ['BENCH_DATA_DIR'],'--checkpoints',str(out/f'h{h}'),
          '--model_id',f'{model}_96_{h}','--model',model,'--data','ETTh1','--features','M',
          '--seq_len','96','--pred_len',str(h),'--enc_in','7','--e_layers','3','--n_heads','4',
          '--d_model','16','--d_ff','128','--dropout','0.1','--des','Exp','--patience','3',
          '--itr','1','--batch_size','32','--learning_rate','0.0001']
    if model=='PatchTST':args+=['--train_epochs','100']
    else:args+=['--backbone','PatchTST','--refine_d_model','128','--refine_d_ff','128',
                '--refine_layers','1','--refine_lr','1e-4','--retrieval_num','50','--refine_epochs','10']
    return args

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','predict']);p.add_argument('--out');p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    cpus=int(os.environ.get('BENCH_CPUS','16'));threads=min(4,cpus)
    horizons=HS
    if a.mode=='predict':
        import numpy as np
        with np.load(a.input,allow_pickle=False) as inp:horizons=tuple(sorted(set(map(int,inp['horizon']))))
    workers=min(len(horizons),4,max(1,cpus//threads))
    target=Path(a.out if a.mode=='train' else a.output).resolve()
    if a.mode=='train':target.mkdir(parents=True,exist_ok=True)
    def worker(slot):
        env=dict(os.environ)
        for key in ['BENCH_CPUS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(threads)
        for h in horizons[slot::workers]:
            if a.mode=='train':
                for m in ['PatchTST','PIR']:
                    with (target/f'{m}_{h}.log').open('w') as log:
                        subprocess.run(command(h,m,target),env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
                    print(f'Completed {m} horizon {h}',flush=True)
            else:
                subprocess.run([sys.executable,'-u',str(CODE/'bench_predict.py'),
                    '--model',a.model,'--input',a.input,'--output',str(target)+f'.h{h}','--horizon',str(h)],env=env,cwd=CODE,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(worker,range(workers)))
    if a.mode=='predict':
        import numpy as np
        arrays={}
        for h in horizons:
            part=Path(str(target)+f'.h{h}')
            with np.load(part,allow_pickle=False) as d:arrays.update({k:d[k] for k in d.files})
            part.unlink()
        with target.open('wb') as f:np.savez(f,**arrays)
if __name__=='__main__':main()
