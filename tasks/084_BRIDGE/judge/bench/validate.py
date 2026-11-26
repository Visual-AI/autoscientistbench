#!/usr/bin/env python3
"""validate.py PRED INPUT; reject invalid tables before scoring."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from table_io import read_table,input_ids

if __name__=='__main__':
    try:
        ids,_=read_table(sys.argv[1],['id','pred'],'pred')
        if ids!=input_ids(sys.argv[2]):raise ValueError('prediction/input id mismatch')
    except (ValueError,KeyError,IndexError,OSError) as e:
        sys.exit(str(e))
    print(f'ok: {len(ids)} rows')
