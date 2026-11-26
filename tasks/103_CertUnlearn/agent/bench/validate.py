#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from prediction import read
if __name__=='__main__':
    try:
        with np.load(sys.argv[2],allow_pickle=False) as f:ids=f['id']
        p,c=read(sys.argv[1],ids)
        if not isinstance(c,dict):raise ValueError('certificate must be an object')
        print('ok: complete finite integer epoch predictions')
    except Exception as e:
        print(str(e),file=sys.stderr);sys.exit(1)
