"""Protocol driver for the four Figure 3(a) methods."""
import argparse,csv,importlib.util,json,os
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from methods import grad_norm_smooth,super_newton

def frozen():
    path=Path(os.environ['BENCH_BENCH_DIR'])/'oracle.py'
    spec=importlib.util.spec_from_file_location('_benchmark_oracle',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','predict'])
    for k in ('out','model','input','output'):p.add_argument('--'+k)
    a=p.parse_args();seed=int(os.environ.get('BENCH_SEED','1'));np.random.seed(seed)
    if a.stage=='train':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        config=dict(n_iters=500,gamma_0=1.0,eps=1e-8,gamma_min=1e-9,H_0=1.0,H_min=1e-5,adaptive_search=True)
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (out/'run.json').write_text(json.dumps({'seed':seed})+'\n')
        return
    config=json.loads((Path(a.model)/'config.json').read_text());api=frozen()
    with np.load(a.input,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'w',newline='') as f,threadpool_limits(limits=min(4,int(os.environ.get('BENCH_CPUS','8')))):
        w=csv.writer(f);w.writerow(['id','method','history','iterations','final_objective','grad_calls','matrix_inverses'])
        for iid,A,b in zip(data['id'],data['A'],data['b']):
            B=A@A.T;Binv=np.linalg.inv(B);x0=np.ones(500)
            for method in api.METHODS:
                oracle=api.make_oracle(A,b,str(iid),method)
                # The original f_star stopping value; the preprocessing query is not a gradient call.
                fstar=float(oracle.func(np.zeros(500)))
                common={k:config[k] for k in ('n_iters','eps','adaptive_search')}
                if method in ('exact_func','inexact_func'):
                    x,status,h=grad_norm_smooth(oracle,x0,B=B,Binv=Binv,f_star=fstar,
                        is_approx=method=='inexact_func',approx_oracle=oracle,
                        approx_hess_fn=lambda o,x:o.approx_hess(x),
                        gamma_0=config['gamma_0'],gamma_min=config['gamma_min'],**common)
                else:
                    x,status,h=super_newton(oracle,x0,B=B,Binv=Binv,f_star=fstar,
                        alpha=1.0 if method.endswith('alpha1') else 0.0,
                        H_0=config['H_0'],H_min=config['H_min'],**common)
                records=oracle.history()
                assert [r['grad_calls'] for r in records]==h['grad_calls']
                assert [r['matrix_inverses'] for r in records]==h['matrix_inverses']
                hit=next((k for k,r in enumerate(records) if r['func']-fstar<=1e-7),None)
                rec=records[hit if hit is not None else -1]
                w.writerow([iid,method,json.dumps(records,separators=(',',':'),allow_nan=False),
                            hit if hit is not None else 500,records[-1]['func'],rec['grad_calls'],rec['matrix_inverses']])
                print(str(iid),method,status,flush=True)

if __name__=='__main__':main()
