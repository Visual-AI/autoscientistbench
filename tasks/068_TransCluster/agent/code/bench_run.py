import argparse,concurrent.futures,importlib.util,json,os,subprocess,sys
from pathlib import Path
from bench_adapter import model_seeds

def train(out):
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((Path(os.environ['BENCH_DATA_DIR'])/'train/generator.json').read_text())
    assert config['steps']==10000
    cpus=int(os.environ['BENCH_CPUS']);threads=min(16,max(1,cpus//3))
    seeds=model_seeds(int(os.environ['BENCH_SEED']))
    def one(pair):
        i,seed=pair;loc=out/f'model_{i}';loc.mkdir(exist_ok=True)
        env=dict(os.environ)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(threads)
        args='--nlb 512 --nub 512 --nclusters 10 --ndims 32 --train_dists normal --scale 0.1 --em --es --sdb --scratch onehot --dqkv 1 --attn_act softmax --attn_itemp 1.0 --loss_act softmax --loss_itemp 10.0 --bsz 32 --init_lr 0.01 --lr_decay 0.5 --patience 5 --nsteps 10000 --nsteps_per_eval 50 --val_nbatches 10'.split()
        with (out/f'train_{i}.log').open('w') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('train.py')),*args,'--seed',str(seed),'--odir',str(loc)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        file=list(loc.glob('*_last.pt'));assert len(file)==1
        print('Completed model',i,'seed',seed,flush=True)
        return str(file[0].relative_to(out))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,cpus)) as pool:files=list(pool.map(one,enumerate(seeds)))
    (out/'models.json').write_text(json.dumps({'files':files,'seeds':seeds,'steps':10000},indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','predict']);ap.add_argument('--out');ap.add_argument('--model');ap.add_argument('--input');ap.add_argument('--output');a=ap.parse_args()
    if a.mode=='train':train(Path(a.out))
    else:
        p=Path(os.environ['BENCH_BENCH_DIR'])/'rollout.py'
        spec=importlib.util.spec_from_file_location('frozen_rollout',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.run(Path(__file__).parent,Path(a.model),Path(a.input),Path(a.output),int(os.environ['BENCH_SEED']),int(os.environ['BENCH_CPUS']))
if __name__=='__main__':main()
