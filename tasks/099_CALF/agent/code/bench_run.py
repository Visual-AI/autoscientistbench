"""Run the four independent author configurations within BENCH_CPUS."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HORIZONS=(96,192,336,720)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    data=Path(os.environ['BENCH_DATA_DIR']).resolve()
    budget=int(os.environ.get('BENCH_CPUS','1'))
    concurrent=min(4,budget);threads=min(16,max(1,budget//concurrent))
    jobs=[];start=time.monotonic()
    for index,h in enumerate(HORIZONS):
        if len(jobs)>=concurrent:
            _,p,f=jobs.pop(0);rc=p.wait();f.close()
            if rc:raise RuntimeError(f'horizon process failed: {rc}')
        env=dict(os.environ)
        env.update({k:str(threads) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        log=(out/f'h{h}.log').open('w')
        cmd=[sys.executable,'-u','run.py','--task_name','long_term_forecast','--is_training','1',
             '--model_id',f'ETTh1_CALF_96_{h}','--model','CALF','--data','ETTh1','--root_path',str(data),
             '--data_path','ETTh1.csv','--features','M','--target','OT','--freq','h','--seq_len','96',
             '--label_len','0','--pred_len',str(h),'--batch_size','256','--learning_rate','0.0005',
             '--lradj','type1','--train_epochs','100','--patience','10','--d_model','768','--n_heads','4',
             '--d_ff','768','--dropout','0.3','--enc_in','7','--dec_in','7','--c_out','7','--gpt_layers','6',
             '--r','8','--lora_alpha','32','--lora_dropout','0.1','--tmax','20','--cos','1',
             '--task_w','1.0','--feature_w','0.01','--output_w','1.0',
             '--word_embedding_path',str(data/'pretrained/wte_pca_500.pt'),
             '--num_workers','0','--itr','1','--checkpoints',str(out/f'h{h}')]
        # No fixed CPU slice per model: on a shared node a busy slice made one horizon 5x slower.
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((h,p,log));print(f'horizon={h} pid={p.pid} threads={threads}',flush=True)
    for h,p,f in jobs:
        rc=p.wait();f.close()
        if rc:raise RuntimeError(f'horizon {h} failed: {rc}; see model directory log')
    print(json.dumps({'train_s':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()
