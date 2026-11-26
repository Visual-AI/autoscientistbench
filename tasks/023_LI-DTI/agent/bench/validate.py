#!/usr/bin/env python3
import sys
from metric import read_rows, align
try:
    pred=read_rows(sys.argv[1], 'pred')
    expected=read_rows(sys.argv[2])
    align(pred,expected)
    print(f'ok: {len(pred)} rows')
except Exception as exc:
    print(f'invalid: {exc}',file=sys.stderr)
    sys.exit(1)
