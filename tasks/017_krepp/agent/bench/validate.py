#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from io_common import input_ids,predictions
try:
    ids,meta=input_ids(sys.argv[2]);predictions(sys.argv[1],ids,meta['edge_count'])
except Exception as e:
    print('invalid prediction: '+str(e),file=sys.stderr);raise SystemExit(1)
print('valid: exactly one placement status per input read')
