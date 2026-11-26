#!/usr/bin/env python3
"""validate.py PRED INPUT: strict schema, finite numbers, ordered id equality."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from format import INPUT, read, aligned
if __name__=='__main__':
    try:
        rows=aligned(sys.argv[1],read(sys.argv[2],INPUT))
        print(f'ok: {len(rows)} rows')
    except Exception as e:
        sys.exit(str(e))
