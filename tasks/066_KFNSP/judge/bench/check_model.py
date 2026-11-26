"""Inductive prediction probe. No automatic harness hook; operator invokes explicitly."""
import argparse,csv,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate import read_predictions

def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
def check(a):
    model=Path(a.model).resolve();code=Path(a.code).resolve()
    before=hashes(model)
    ids,values=read_predictions(a.pred);baseline=dict(zip(ids,values))
    with open(a.input,newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    with tempfile.TemporaryDirectory() as tmp:
        # Deterministic sparse selection and reversed order check prediction independence from batch context.
        selected=rows[::max(1,len(rows)//37)][::-1]
        inp=Path(tmp)/'input.csv';out=Path(tmp)/'pred.csv'
        with inp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(selected)
        subprocess.run(['bash',str(code/'predict.sh'),'--model',str(model),'--input',str(inp),'--output',str(out)],check=True)
        got,pred=read_predictions(out)
        if got!=[r['id'] for r in selected]:raise ValueError('probe id mismatch')
        err=max(abs(v-baseline[i]) for i,v in zip(got,pred))
        if err>1e-6:raise ValueError(f'batch-dependent predictions: max error {err}')
    if before!=hashes(model):raise ValueError('prediction changed model artifacts')
    return {'pass':True,'max_abs_error':err,'model_unchanged':True}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model');ap.add_argument('--code',required=True);ap.add_argument('--input',required=True);ap.add_argument('--pred',required=True);a=ap.parse_args()
    try:report=check(a)
    except Exception as e:report={'pass':False,'fair_mae':float('inf'),'reason':str(e)}
    print(json.dumps(report));sys.exit(0 if report['pass'] else 1)
