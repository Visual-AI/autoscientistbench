"""CSV schema validation. Usage: validate.py prediction input."""
import csv,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dueling_env import T,K


def read_pred(path):
    with open(path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','actions','rounds','regret']:raise ValueError('Wrong prediction header')
        result={}
        for row in r:
            if None in row or any(x is None for x in row.values()):raise ValueError('Malformed row')
            key=row['id']
            if not key or key in result:raise ValueError('Empty or duplicate id')
            actions=json.loads(row['actions']);n=int(row['rounds']);regret=float(row['regret'])
            if not math.isfinite(regret) or regret<0:raise ValueError('Regret must be finite and nonnegative')
            if n!=T or not isinstance(actions,list) or len(actions)!=T:raise ValueError('Full horizon required')
            if any(not isinstance(p,list) or len(p)!=2 or any(type(a)!=int or not 0<=a<K for a in p) for p in actions):raise ValueError('Invalid actions')
            result[key]=(actions,n,regret)
    if not result:raise ValueError('Empty predictions')
    return result


def main(pred,inp):
    with open(inp,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','seed']:raise ValueError('Wrong input header')
        ids=[x['id'] for x in r]
    rows=read_pred(pred)
    if len(set(ids))!=len(ids) or set(ids)!=set(rows):raise ValueError('ID set mismatch')
    print(f'ok: {len(rows)} episodes')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
