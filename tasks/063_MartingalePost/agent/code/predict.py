import argparse,csv,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
assert Path(a.model,'ready').exists()
tmp=Path(a.output).with_suffix('.csv');start=time.monotonic()
subprocess.run(['julia','--startup-file=no','--compiled-modules=no','method.jl',a.input,str(tmp)],check=True)
elapsed=time.monotonic()-start
rows=[]
with tmp.open(newline='') as f:
 for r in csv.DictReader(f):rows.append({'id':r['id'],'lo':[float(r['lo'+str(j)]) for j in [1,3,5]],'hi':[float(r['hi'+str(j)]) for j in [1,3,5]]})
Path(a.output).write_text(json.dumps({'elapsed_s':elapsed,'predictions':rows},allow_nan=False)+'\n')
