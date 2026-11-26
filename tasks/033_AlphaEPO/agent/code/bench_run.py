"""Protocol entry points for the unchanged reproduced pair approximation."""
import argparse,csv,json,os,random,resource,subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='mode',required=True)
    t=sub.add_parser('train');t.add_argument('--out',required=True)
    p=sub.add_parser('predict');p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=ap.parse_args();seed=int(os.environ.get('BENCH_SEED','1'));cpus=int(os.environ.get('BENCH_CPUS','16'));assert cpus>=1
    random.seed(seed)
    if args.mode=='train':
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(Path(__file__).parent/'pa_rhs.cpp'),'-o',str(out/'pa_rhs.so')],check=True)
        config={'k':14.81,'alpha':.1,'q':3,'end_time':2000.,'rtol':1e-7,'atol':1e-9,'ordered_IC':True,'solver':'RK45','seed':seed}
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        return
    model=Path(args.model);config=json.loads((model/'config.json').read_text())
    os.environ['PA_LIBRARY']=str(model/'pa_rhs.so')
    import numpy as np
    np.random.seed(seed)
    from pa_predict import curve
    with open(args.input,newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['id','p'];rows=list(reader)
    assert len({r['id'] for r in rows})==len(rows) and rows
    # Points start independently at ordered IC; sorting only canonicalizes execution.
    ps=sorted({float(r['p']) for r in rows});assert all(0<=p<=.5 for p in ps)
    arr,_=curve(config['k'],pgrid=np.array(ps),rtol=config['rtol'],atol=config['atol'],alpha=config['alpha'])
    values=dict(zip(ps,arr[:,1]))
    usage=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu=usage.ru_utime+usage.ru_stime+child.ru_utime+child.ru_stime
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    with open(args.output,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','pred','cpu_s','analytic'])
        w.writerows([(r['id'],repr(float(values[float(r['p'])])),repr(cpu),1) for r in rows])
    print(json.dumps({'cpu_s_before_output':cpu,'points':len(ps),'cpus_budget':cpus}))
if __name__=='__main__':main()
