#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from network import read_prediction

if __name__=='__main__':
    try:
        with np.load(sys.argv[2],allow_pickle=False) as data: ids=data['ids'].tolist()
        records=read_prediction(sys.argv[1],ids)
        print(f'ok: {len(records)} TTN certificates')
    except Exception as e:
        print(str(e),file=sys.stderr);sys.exit(1)
