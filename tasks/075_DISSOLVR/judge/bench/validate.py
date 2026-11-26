#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from table import read,prediction

def main(p,x):
 pred=prediction(p);inp=read(x,['id','SMILES'])
 if [r['id'] for r in pred]!=[r['id'] for r in inp]:raise ValueError('prediction/input ids differ')
 print(f'ok: {len(pred)} rows, five finite prediction columns')
if __name__=='__main__':
 try:main(*sys.argv[1:])
 except Exception as e:sys.exit(str(e))
