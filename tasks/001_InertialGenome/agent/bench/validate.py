"""Strict padded-coordinate CSV validation, independent of candidate code."""
import csv
import sys
import numpy as np

MAX_LEN=2400

def read_prediction(path,ids):
    expected=[(str(i),str(slot)) for i in ids for slot in range(MAX_LEN)]
    xyz=[];seen=set()
    with open(path,newline='') as f:
        reader=csv.reader(f)
        if next(reader,None)!=['id','slot','x','y','z']:raise ValueError('header must be id,slot,x,y,z')
        for index,row in enumerate(reader):
            if len(row)!=5:raise ValueError('expected five columns')
            key=(row[0],row[1])
            if key in seen:raise ValueError('duplicate id/slot')
            if index>=len(expected) or key!=expected[index]:raise ValueError('id/slot set or order mismatch')
            seen.add(key);values=np.asarray(row[2:],dtype=np.float32)
            if not np.isfinite(values).all():raise ValueError('nonfinite coordinate')
            xyz.append(values)
    if len(xyz)!=len(expected):raise ValueError('missing coordinates')
    return np.asarray(xyz,dtype=np.float32).reshape(len(ids),MAX_LEN,3)

def main(pred,inp):
    with np.load(inp,allow_pickle=False) as data:
        ids=data['ids'].tolist()
    if len(set(ids))!=len(ids):raise ValueError('duplicate input id')
    read_prediction(pred,ids)
    print('ok')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
