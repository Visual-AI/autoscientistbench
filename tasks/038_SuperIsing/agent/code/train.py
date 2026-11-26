"""Persistent ED workers, one per lz2 sector, and SciPy's standard Nelder–Mead (paper cost)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scipy.optimize import minimize
from eval_cost import cost, atomic_json

root = Path(__file__).resolve().parent
out = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv)>2 else 'full'
t0 = time.monotonic()
# Sector solves are independent. ARPACK is not thread-safe, so each sector gets
# its own process; serial BLAS keeps the total at BENCH_CPUS threads.
threads = str(max(1,int(os.environ['BENCH_CPUS'])//2))
workers = [subprocess.Popen(['julia', '--compiled-modules=no', '--startup-file=no', '-t', threads, '--project='+str(root/'env'), str(root/'run_super_ising.jl')], env={**os.environ,'ED_SECTOR':str(lz),'OMP_NUM_THREADS':threads,'OPENBLAS_NUM_THREADS':'1'}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1) for lz in (0,1)]
manifest = dict(fixed_couplings=[1.5,.25,1.0], observable_convention='ObsMomIncr=false (original example convention; required nonzero pair conversion)',cases=[],local_ed_evaluations=0,reductions=[], optimiser='scipy.optimize.minimize Nelder-Mead, adaptive=False, default 5% initial simplex, xatol=1e-4,fatol=1e-4', stage_budget_s=3600)

def wait_for(prefix):
    for worker in workers:
        for line in worker.stdout:
            print(line, end='',flush=True)
            if line.startswith(prefix): break
        else:
            raise RuntimeError(f'Julia exited before {prefix}, rc={worker.poll()}')

def merge_sectors(path):
    # Same rows and order as one process: stable sort by E, lz2=0 rows first.
    parts = [Path(f'{path}.lz{lz}') for lz in (0,1)]
    head,*rows = parts[0].read_text().splitlines()
    rows += parts[1].read_text().splitlines()[1:]
    rows.sort(key=lambda r:float(r.split(',')[0]))
    path.write_text('\n'.join([head,*rows])+'\n')
    for part in parts: part.unlink()

def run(n,tag,params):
    f = f'N{n}_{tag}.csv'
    start=time.monotonic()
    for lz,worker in enumerate(workers):
        worker.stdin.write('\t'.join(map(str,[n,*params,f'{out/f}.lz{lz}']))+'\n'); worker.stdin.flush()
    wait_for('DONE\t')
    merge_sectors(out/f)
    measurement=cost(out/f) if mode!='smoke' else dict(cost=0.0,lambda_opt=0.0)
    item=dict(nmf=n,tag=tag,parameters=list(map(float,params)),file=f,training_cost=measurement['cost'],wall_s=time.monotonic()-start)
    manifest['cases'].append(item)
    if tag.startswith('local'): manifest['local_ed_evaluations']+=1
    manifest['elapsed_s']=time.monotonic()-t0
    atomic_json(out/'simulation.json',manifest)
    print(f"COST {tag} N={n} cost={measurement['cost']:.10f} lambda={measurement['lambda_opt']:.12f} elapsed={manifest['elapsed_s']:.1f}s",flush=True)
    return measurement['cost']

try:
    wait_for('READY')
    if mode=='smoke':
        run(4,'smoke',[.0680,.0809,.0776])
    elif mode=='pilot':
        run(9,'pilot9',[.0680,.0809,.0776])
    else:
        for n,p in [(6,[.0662,.0793,.0615]),(7,[.0698,.0774,.0722]),(8,[.0690,.0792,.0757])]:
            run(n,f'table{n}',p)
        run(9,'table9',[.0680,.0809,.0776])
        run(9,'example9',[.0698080753241176,.0773934248387392,.07224789112627583])
        best=min((r for r in manifest['cases'] if r['nmf']==9),key=lambda r:r['training_cost'])
        # Fixed at 20: choosing 40/20/0 from measured ED time made the method,
        # and so the cost, depend on node speed.
        per_eval=manifest['cases'][-1]['wall_s']
        remaining=3500-(time.monotonic()-t0)
        count=20
        manifest['reductions'].append(f'Local Nelder-Mead limited to {count} ED evaluations (fixed, not timing-based): measured cached ED {per_eval:.2f}s, remaining budget {remaining:.2f}s.')
        print(f'OPTIMISATION maxfev={count}, cached ED={per_eval:.2f}s, remaining={remaining:.2f}s',flush=True)
        def objective(p):
            return run(9,f"local{manifest['local_ed_evaluations']+1:02d}",p)
        opt=minimize(objective,best['parameters'],method='Nelder-Mead',options=dict(maxfev=count,xatol=1e-4,fatol=1e-4,adaptive=False))
        manifest['optimisation_result']=dict(nfev=int(opt.nfev),nit=int(opt.nit),message=str(opt.message),success=bool(opt.success),x=opt.x.tolist(),fun=float(opt.fun))
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.stdin.write('QUIT\n'); worker.stdin.flush()
            worker.stdin.close()
            for line in worker.stdout: print(line,end='',flush=True)
    rcs=[worker.wait() for worker in workers]
    manifest['elapsed_s']=time.monotonic()-t0
    atomic_json(out/'simulation.json',manifest)
    if any(rcs): raise RuntimeError(f'Julia exit {rcs}')
