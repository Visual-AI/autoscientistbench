#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pulse_io import instances, predictions

def main(pred,inputs):
    rows=instances(inputs)
    predictions(pred,[r['id'] for r in rows])
    print('ok: pulse rows and finite parameters')
if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception as exc:sys.exit(str(exc))
