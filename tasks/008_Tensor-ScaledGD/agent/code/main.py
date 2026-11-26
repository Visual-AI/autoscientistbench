"""Prepare the original numerical runtime/configuration and launch solver workers."""
import argparse,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
from runtime import prepare,environment
DEFAULT=dict(eta=0.5,eta_rtc=0.4,z0=0.5,z1=0.5,decay=0.95)

def launch(script,runtime,cpus,parallel,log):
    env=environment(runtime,len(cpus) if parallel else min(2,len(cpus)),parallel)
    env['REPRO_OCTAVE_PRELOAD']=env.pop('LD_PRELOAD','')
    cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,str(Path(__file__).parent/'cpu_opt/bind_exec.py'),'--no-bind',str(runtime/'octave/bin/octave-cli'),'--no-history','--no-init-file','--quiet',str(script)]
    with open(log,'w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train']);parser.add_argument('--out',required=True);args=parser.parse_args()
    random.seed(int(os.environ['BENCH_SEED']));np.random.seed(int(os.environ['BENCH_SEED']))
    started=time.monotonic();model=Path(args.out);model.mkdir(parents=True,exist_ok=True);prepare(model)
    (model/'config.json').write_text(json.dumps(DEFAULT)+'\n')
    (model/'train_seconds.json').write_text(json.dumps(time.monotonic()-started)+'\n')
if __name__=='__main__':main()
