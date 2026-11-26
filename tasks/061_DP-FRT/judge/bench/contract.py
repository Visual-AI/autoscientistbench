import csv
import math

def read_predictions(path):
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['id','lo','hi','credible_set']: raise ValueError('prediction columns')
        out={}
        for r in reader:
            if not r['id'] or r['id'] in out: raise ValueError('duplicate or empty id')
            lo,hi=float(r['lo']),float(r['hi'])
            values=[float(v) for v in r['credible_set'].split(';')]
            if not all(math.isfinite(v) for v in [lo,hi,*values]): raise ValueError('nonfinite')
            if not 0 <= lo <= hi <= 1: raise ValueError('invalid interval')
            if not values or any(a>=b for a,b in zip(values,values[1:])): raise ValueError('invalid set')
            if any(v<lo-1e-12 or v>hi+1e-12 for v in values): raise ValueError('set outside interval')
            if abs(values[0]-lo)>1e-12 or abs(values[-1]-hi)>1e-12: raise ValueError('set endpoints differ')
            out[r['id']]={'lo':lo,'hi':hi,'values':set(round(v,12) for v in values)}
    if not out: raise ValueError('empty output')
    return out
