"""Time the complete training driver and save prediction metadata."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
code=Path(__file__).resolve().parent;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
seed=int(os.environ.get('BENCH_SEED','1'))
t0=time.perf_counter()
subprocess.run([sys.executable,str(code/'train_lshape_param.py'),
 '--repo',str(code/'2D/L_shape/parametric'),'--data-dir',os.environ['BENCH_DATA_DIR'],
 '--out-dir',str(out),'--n','10','--steps','1400000','--lr','0.01',
 '--lr-schedule','fixed','--impl','fast','--log-every','10000',
 '--seed',str(seed),'--sample-seed',str(seed+58)],check=True)
train_s=time.perf_counter()-t0
card={'format':'rfem-run/v1','seed':seed,'sample_seed':seed+58,'steps':1400000,
 'train_s':train_s,'weights_sha256':hashlib.sha256((out/'trained_params.npz').read_bytes()).hexdigest()}
(out/'model_card.json').write_text(json.dumps(card,indent=2)+'\n')
print(json.dumps(card),flush=True)
