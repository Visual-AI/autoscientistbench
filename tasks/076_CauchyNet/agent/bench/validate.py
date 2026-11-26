#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import predictions,read_csv

def main(pred_path,input_path):
    ids,_,_=predictions(pred_path)
    inp=read_csv(input_path,['id','x'])
    if ids!=[r['id'] for r in inp]:raise ValueError('ids/order must exactly match input')
    print(f'ok: {len(ids)} rows, ten predictions per row')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
