"""Strict CSV schema, complete finite trajectory, and input ID/order validator."""
import csv
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clasp_env import HEADER, N, T


def read_pred(path):
    records = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValueError('prediction header must be '+','.join(HEADER))
        for r in reader:
            if None in r or any(v is None for v in r.values()):
                raise ValueError('wrong column count')
            actions = json.loads(r['actions'])
            if not isinstance(actions, list) or len(actions) != T+1:
                raise ValueError('trajectory must contain T+1 decisions')
            if any(not isinstance(x,list) or len(x)!=N for x in actions):
                raise ValueError('each decision must have N coordinates')
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for x in actions for v in x):
                raise ValueError('nonfinite or nonnumeric action')
            if int(r['rounds']) != T or r['rounds'] != str(T):
                raise ValueError('rounds must be T')
            for name in ('loss','ccv1','ccv2'):
                v=float(r[name])
                if not math.isfinite(v) or v < 0:
                    raise ValueError('metric must be finite and nonnegative')
                r[name]=v
            r['actions']=actions; records.append(r)
    ids=[r['id'] for r in records]
    if not records or len(set(ids))!=len(ids) or any(not i for i in ids):
        raise ValueError('empty predictions or duplicate/empty IDs')
    return records


def main(pred, inp):
    records=read_pred(pred)
    with open(inp,newline='') as f:ids=[r['id'] for r in csv.DictReader(f)]
    if [r['id'] for r in records]!=ids:
        raise ValueError('IDs must match the input exactly, including order')
    print(f'ok: {len(ids)} complete trajectories')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except (ValueError,KeyError,TypeError,OSError) as exc:sys.exit(str(exc))
