"""Strict CSV and nested numerical trajectory validation."""
import csv,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from oracle import METHODS
csv.field_size_limit(100_000_000)
HEADER=['id','method','history','iterations','final_objective','grad_calls','matrix_inverses']

def read_predictions(path,ids):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=HEADER: raise ValueError('wrong CSV header')
        rows=list(reader)
    expected={(str(i),m) for i in ids for m in METHODS}
    seen=set();out={}
    for r in rows:
        if None in r or any(v is None for v in r.values()):raise ValueError('row width')
        key=(r['id'],r['method'])
        if key in seen: raise ValueError('duplicate id/method')
        seen.add(key)
        for k in HEADER[3:]:
            v=float(r[k])
            if not math.isfinite(v):raise ValueError('nonfinite summary')
            if k!='final_objective' and (v<0 or v!=int(v)):raise ValueError('invalid integer')
            r[k]=v
        h=json.loads(r['history'])
        if not isinstance(h,list) or not 1<=len(h)<=501:raise ValueError('history length')
        for rec in h:
            if set(rec)!={'x','func','grad_calls','matrix_inverses','events'}:raise ValueError('history keys')
            x=np.asarray(rec['x'],dtype=float)
            if x.shape!=(500,) or not np.isfinite(x).all():raise ValueError('iterate shape/nonfinite')
            if not math.isfinite(float(rec['func'])):raise ValueError('nonfinite objective')
            for c in ('grad_calls','matrix_inverses'):
                v=rec[c]
                if type(v) is not int or v<0:raise ValueError('invalid history counter')
            if not isinstance(rec['events'],list) or any(e not in ('func','grad','hess','approx_hess','inverse') for e in rec['events']):raise ValueError('invalid events')
        r['history']=h;out[key]=r
    if seen!=expected:raise ValueError('id/method set mismatch')
    return out

def main(pred,inp):
    with np.load(inp,allow_pickle=False) as z: ids=z['id']
    read_predictions(pred,ids)
    print('ok')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
