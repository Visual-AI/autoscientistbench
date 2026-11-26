"""Protocol driver for the unchanged Figure-4 AsymP-DGDA algorithm."""
import argparse,csv,importlib.util,json,os,random,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def frozen_env():
    path=Path(os.environ['BENCH_BENCH_DIR'])/'leduc_env.py'
    spec=importlib.util.spec_from_file_location('_counted_leduc',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def worker(args):
    path,iid,player,config=args
    import numpy as np
    import LiteEFG
    from algorithm.asymp_dgda import _AsymPDGDA
    seed=int(os.environ.get('BENCH_SEED','1'))
    random.seed(seed);np.random.seed(seed);LiteEFG.set_seed(seed)
    mus=[0.,0.];mus[player]=config['mu']
    graph=_AsymPDGDA(eta=config['eta'],mus=mus)
    env=frozen_env().open_run(path,iid,graph,player)
    start=time.monotonic()
    for i in range(1,config['steps_per_run']+1):
        graph.update_graph(env)
        if i%10000==0:
            print(json.dumps({'instance':iid,'run':player,'step':i,'elapsed_s':time.monotonic()-start}),flush=True)
    state=env.export(graph.u)
    return iid,player,state


def train(out):
    # No fitting across games: save the original reproduction's configuration.
    out.mkdir(parents=True,exist_ok=True)
    config={'eta':.1,'mu':.0001,'steps_per_run':500000,'seed':int(os.environ.get('BENCH_SEED','1'))}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')


def predict(model,input_path,output):
    config=json.loads((model/'config.json').read_text())
    env=frozen_env();ids=env.instances(input_path)
    jobs=[(str(input_path),iid,p,config) for iid in ids for p in (0,1)]
    cpus=max(1,int(os.environ.get('BENCH_CPUS','1')))
    if cpus>1:
        with ProcessPoolExecutor(max_workers=min(cpus,len(jobs))) as pool:states=list(pool.map(worker,jobs))
    else:states=list(map(worker,jobs))
    states={(iid,p):state for iid,p,state in states}
    with output.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','instance','player','infoset','action','probability','updates','traversals','oracle'])
        for iid in ids:
            pair=[states[(iid,p)] for p in (0,1)]
            updates=sum(s['updates'] for s in pair);traversals=sum(s['traversals'] for s in pair)
            oracle=min(s['oracle'] for s in pair)
            # Exactly the source's combination: player p from the run perturbing p.
            for p in (0,1):
                for j,(_,values) in enumerate(pair[p]['players'][p]):
                    for a,value in enumerate(values):
                        w.writerow([f'{iid}:{p+1}:{j}:{a}',iid,p+1,j,a,repr(value),updates,traversals,oracle])


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('train');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('predict')
    for arg in ('model','input','output'):p.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='train':train(args.out)
    else:predict(args.model,args.input,args.output)
