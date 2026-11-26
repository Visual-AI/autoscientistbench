import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from format import read_predictions

def validate(pred,inp):
    ids,p=read_predictions(pred)
    with np.load(inp,allow_pickle=False) as z:
        if set(z.files)!={'id','x'}:raise ValueError('input keys')
        if z['x'].shape!=(len(ids),512) or not np.isfinite(z['x']).all():raise ValueError('input shape/values')
        if not np.array_equal(ids,z['id']):raise ValueError('prediction/input IDs mismatch')
if __name__=='__main__':
    try:validate(*sys.argv[1:]);print('OK')
    except Exception as e:print(str(e),file=sys.stderr);sys.exit(1)
