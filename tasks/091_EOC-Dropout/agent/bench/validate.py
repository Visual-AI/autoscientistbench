"""Usage: validate.py PRED INPUT; shape, IDs, and finite output validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from schema import load

def main(pred,inp):
    d=load(pred)
    with np.load(inp,allow_pickle=False) as z:
        ids=z['ids'];x=z['x']
        if set(z.files)!={'ids','x'} or x.shape!=(len(ids),3,32,32) or not np.isfinite(x).all():raise ValueError('invalid input')
    if not np.array_equal(d['ids'],ids):raise ValueError('id set/order mismatch')
    print('ok')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
