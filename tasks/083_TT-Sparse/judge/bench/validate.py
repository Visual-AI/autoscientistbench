#!/usr/bin/env python3
"""Validate CSV ids, five finite probabilities, and the JSON rule certificate."""
import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from certificate import read_prediction, FEATURES

def main(pred_path,input_path):
    ids,_,_=read_prediction(pred_path)
    with open(input_path,newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['id',*FEATURES],'input header mismatch'
        expected=[r['id'] for r in reader]
    assert len(set(expected))==len(expected) and set(ids)==set(expected),'id set mismatch'
    print(f'ok: {len(ids)} rows')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(f'invalid: {e}')
