"""Train five independent APGC runs; expose their transductive assignments."""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

def train(a):
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cpus=int(os.environ['BENCH_CPUS']);outer=int(os.environ['BENCH_SEED'])
    threads=max(1,min(3,cpus//5));parallel=max(1,min(5,cpus//threads))
    seeds=[2023+5*(outer-1)+i for i in range(5)]
    pending=list(enumerate(seeds));running=[];summary=[]
    try:
        while pending or running:
            while pending and len(running)<parallel:
                slot,seed=pending.pop(0);d=out/f'run_{slot}';d.mkdir(exist_ok=True)
                e=os.environ.copy();e.update(APGC_INNER_SEED=str(seed),APGC_OUTPUT_DIR=str(d))
                for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:e[key]=str(threads)
                log=(d/'train.log').open('w')
                p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('train.py'))],env=e,stdout=log,stderr=subprocess.STDOUT)
                running.append((slot,seed,p,log,time.monotonic()))
            for item in list(running):
                slot,seed,p,log,t=item
                if p.poll() is not None:
                    log.close();running.remove(item)
                    if p.returncode:raise RuntimeError(f'run {slot} failed; see its train.log')
                    row={'slot':slot,'seed':seed,'wall_s':time.monotonic()-t};summary.append(row);print(json.dumps(row),flush=True)
            if running:time.sleep(0.2)
    finally:
        for _,_,p,log,_ in running:
            p.terminate();p.wait();log.close()
    (out/'runs.json').write_text(json.dumps({'outer_seed':outer,'threads_per_run':threads,'runs':sorted(summary,key=lambda r:r['slot'])},indent=2)+'\n')

def predict(a):
    import torch
    with open(a.input,newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['id']
        ids=[r['id'] for r in reader]
    assert len(ids)==len(set(ids))
    columns=[]
    for slot in range(5):
        ck=torch.load(Path(a.model)/f'run_{slot}'/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        by_id=dict(zip(ck['node_ids'],ck['predictions']))
        columns.append([int(by_id[i]) for i in ids])
    with open(a.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id']+[f'pred_{i}' for i in range(5)])
        w.writerows([i]+[p[j] for p in columns] for j,i in enumerate(ids))

if __name__=='__main__':
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    t=sp.add_parser('train');t.add_argument('--out',required=True)
    p=sp.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=ap.parse_args();train(a) if a.cmd=='train' else predict(a)
