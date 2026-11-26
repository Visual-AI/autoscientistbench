"""Strict CSV interval output validation; input is a collection of dataset arrays."""
import csv,sys
import numpy as np
HEADER=['id','pred','lower','upper']
def read_predictions(path):
    out={}
    with open(path,newline='') as f:
        reader=csv.reader(f)
        if next(reader,None)!=HEADER:raise ValueError('header must be id,pred,lower,upper')
        for row in reader:
            if len(row)!=4:raise ValueError('wrong column count')
            id_=row[0]
            if not id_ or id_ in out:raise ValueError('empty or duplicate id')
            a=np.asarray(row[1:],dtype=np.float64)
            if not np.isfinite(a).all():raise ValueError('nonfinite number')
            if not a[1]<=a[0]<=a[2]:raise ValueError('require lower <= pred <= upper')
            out[id_]=a
    return out

def main(pred_path,input_path):
    pred=read_predictions(pred_path)
    with np.load(input_path,allow_pickle=False) as inp:
        expected={f'{id_}_r{r}' for key in inp.files if key.startswith('id_') for id_ in inp[key] for r in range(10)}
    if set(pred)!=expected:raise ValueError('id set mismatch')
    print('ok:',len(pred),'rows')
if __name__=='__main__':
    try:main(sys.argv[1],sys.argv[2])
    except Exception as e:sys.exit(str(e))
