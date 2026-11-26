"""Accept parameter schedules only; recompute spectra with frozen physics code."""
import json,os,subprocess,sys,tempfile,time,csv,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read
from eval_cost import cost
def score(pred_path,labels_path):
 labels=json.loads(Path(labels_path).read_text());p=read(pred_path,labels['mode']);base=Path(__file__).resolve().parent
 scratch=Path(tempfile.mkdtemp(prefix='phy18-ed-'));commands=[];paths=[]
 for i,r in enumerate(p['cases']):
  target=scratch/f'case{i}.csv';paths.append(target);commands.append('\t'.join(map(str,[r['nmf'],*r['parameters'],target])))
 commands.append('QUIT');start=time.monotonic()
 with (scratch/'worker.log').open('w') as log:
  result=subprocess.run(['julia','--startup-file=no','--compiled-modules=no','-t',os.environ.get('BENCH_CPUS','1'),'--project='+str(base/'env'),str(base/'run_super_ising.jl')],input='\n'.join(commands)+'\n',text=True,stdout=log,stderr=subprocess.STDOUT,timeout=None)
 if result.returncode:raise RuntimeError('Frozen ED failed: '+(scratch/'worker.log').read_text()[-2500:])
 if labels['mode']=='smoke':
  with paths[0].open() as f:rows=list(csv.DictReader(f))
  residual=max(float(r['residual']) for r in rows)
  assert len(rows)==40 and residual<1e-6 and all(math.isfinite(float(r['E'])) for r in rows)
  return {'cost':1.0,'constraints_pass':False,'smoke_only':True,'actual_ed_nmf':4,'states':len(rows),'max_eigen_residual':residual,'ed_wall_s':time.monotonic()-start,'reason':'Full N9 protocol deliberately not run in smoke.'}
 values=[{**r,**cost(path)} for r,path in zip(p['cases'],paths)]
 best=min((r for r in values if r['nmf']==9),key=lambda r:r['cost'])
 return {'cost':best['cost'],'constraints_pass':True,'best_tag':best['tag'],'best_parameters':best['parameters'],'lambda_opt':best['lambda_opt'],'operator_count':14,'states':40,'local_ed_evaluations':len(values)-5,'ed_wall_s':time.monotonic()-start,'constraint_scope':'The three explicit reproduction protocol checks; unspecified external constraints cannot be certified.'}
if __name__=='__main__':print(json.dumps(score(*sys.argv[1:3]),allow_nan=False))
