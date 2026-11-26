#!/usr/bin/env python3
import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from format import read_predictions

def main(pred,input_path):
    p=read_predictions(pred)
    with open(input_path,newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id']:raise ValueError('input header must be id')
        ids=[row['id'] for row in r]
    if len(ids)!=len(set(ids)) or set(ids)!=set(p):raise ValueError('id set mismatch')
    if list(p)!=ids:raise ValueError('rows must follow input order')
    print(f'ok: {len(p)} rows, five assignments')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
