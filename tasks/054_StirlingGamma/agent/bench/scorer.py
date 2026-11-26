#!/usr/bin/env python3
import json
import math
import sys
from format import read, predictions
FLOORS={2:0.835,4:0.550,6:0.458}
def score(prediction,labels):
    truth=labels['datasets']
    ids=[r['id'] for r in truth]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate label IDs")
    pred=predictions(prediction,ids)
    cells={k:[] for k in FLOORS}
    for row in truth:
        k=row['kstar']
        if type(k) is not int or k not in cells:
            raise ValueError("unexpected true K")
        cells[k].append(pred[row['id']][k-1])
    if any(len(v)==0 for v in cells.values()) or len({len(v) for v in cells.values()})!=1:
        raise ValueError("all three equally sized K cells required")
    means={k:math.fsum(v)/len(v) for k,v in cells.items()}
    passed=all(means[k]>=floor for k,floor in FLOORS.items())
    return {'posterior_probability':math.fsum(means.values())/3 if passed else 0.0,
            'cell_means':{str(k):v for k,v in means.items()},'cell_floor_guard_passed':passed}
if __name__=='__main__':
    try:
        result=score(read(sys.argv[1]),read(sys.argv[2]))
    except Exception as exc:
        result={'posterior_probability':0.0,'invalid':str(exc)}
    print(json.dumps(result,allow_nan=False))
