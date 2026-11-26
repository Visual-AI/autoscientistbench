"""Strict CSV schema, finite values, exact id sequence; exit nonzero on rejection."""
import csv,math,sys

def read_predictions(path):
    with open(path,newline='') as f:
        r=csv.reader(f)
        if next(r,None)!=['id','pred']:raise ValueError('header must be exactly id,pred')
        ids=[];values=[]
        for row in r:
            if len(row)!=2 or not row[0]:raise ValueError('each row needs id,pred')
            v=float(row[1])
            if not math.isfinite(v):raise ValueError('prediction must be finite')
            ids.append(row[0]);values.append(v)
        if not ids or len(set(ids))!=len(ids):raise ValueError('empty output or duplicate ids')
    return ids,values

def main(pred,input_path):
    ids,_=read_predictions(pred)
    with open(input_path,newline='') as f:expected=[r['id'] for r in csv.DictReader(f)]
    if ids!=expected or len(set(expected))!=len(expected):raise ValueError('id set or order mismatch')
    print(f'ok: {len(ids)} rows')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
