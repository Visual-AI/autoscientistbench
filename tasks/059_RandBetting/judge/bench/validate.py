"""Strict prediction table validation. Row order is immaterial; ID set is exact."""
import csv,json,math,sys,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import betting_interface as bi
COLS=['id','e_value','u','reject','policy','seed']

def read_pred(path):
    rows={};cert=None;seed=None
    with open(path,newline='') as f:
        rd=csv.DictReader(f)
        if rd.fieldnames!=COLS:raise ValueError('wrong header')
        for r in rd:
            if None in r or any(v is None for v in r.values()):raise ValueError('column count')
            if not r['id'] or r['id'] in rows:raise ValueError('empty or duplicate id')
            vals=[float(r[k]) for k in ('e_value','u','reject','seed')]
            if not all(math.isfinite(v) for v in vals):raise ValueError('nonfinite value')
            e,u,reject,s=vals
            if e<0 or not 0<=u<1 or reject not in (0,1) or s!=int(s) or not 0<=s<2**32:raise ValueError('range')
            p=bi.canonical(json.loads(r['policy']))
            if cert is not None and (p!=cert or s!=seed):raise ValueError('policy and seed must be global')
            cert=p;seed=s
            rows[r['id']]=(e,u,int(reject))
    if not rows:raise ValueError('empty prediction')
    return rows,json.loads(cert),int(seed)

def main(pred,inp):
    if zipfile.is_zipfile(pred):
        from private_betting import REQUEST_PROTOCOL,validate_package
        if json.loads(Path(inp).read_text())!={'protocol':REQUEST_PROTOCOL}:
            raise ValueError('invalid private betting request')
        _,method=validate_package(pred)
        print('ok: private betting artifact ('+method['execution']+')')
        return
    rows,_,_=read_pred(pred)
    with np.load(inp,allow_pickle=False) as z:ids=[str(i) for i in z['id']]
    if len(ids)!=len(set(ids)) or set(rows)!=set(ids):raise ValueError('id set mismatch')
    print(f'ok: {len(rows)} predictions')
if __name__=='__main__':
    try:
        if len(sys.argv)!=3:raise ValueError('usage: validate.py PRED INPUT')
        main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
