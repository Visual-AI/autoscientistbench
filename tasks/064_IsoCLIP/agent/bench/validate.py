#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_prediction

def main(pred,inp):
    shapes={}
    for path in sorted(Path(inp).glob('*.npz')):
        with np.load(path,allow_pickle=False) as d:shapes[path.stem]=(len(d['query']),len(d['gallery']))
    if not shapes:raise ValueError('empty input')
    load_prediction(pred,shapes)
    print('ok: complete finite similarity matrices')

if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as e:sys.exit(str(e))
